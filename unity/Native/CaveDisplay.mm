#import <AppKit/AppKit.h>
#import <QuartzCore/CAMetalLayer.h>

// Use Metal's own display synchronization without Unity's macOS VSync wait path.
// Only visits layers owned by this process; no global display settings change.
static CAMetalLayer *FindMetalLayer(CALayer *layer) {
    if ([layer isKindOfClass:[CAMetalLayer class]]) return (CAMetalLayer *)layer;
    for (CALayer *child in layer.sublayers) {
        CAMetalLayer *found = FindMetalLayer(child);
        if (found) return found;
    }
    return nil;
}
static CAMetalLayer *FindViewLayer(NSView *view) {
    CAMetalLayer *found = FindMetalLayer(view.layer);
    if (found) return found;
    for (NSView *child in view.subviews) {
        found = FindViewLayer(child);
        if (found) return found;
    }
    return nil;
}
extern "C" int CaveSetDisplaySync(int enabled) {
    if (![NSThread isMainThread]) return 0;
    int result = 0;
    for (NSWindow *window in NSApp.windows) {
        CAMetalLayer *layer = FindViewLayer(window.contentView);
        if (!layer) continue;
        layer.displaySyncEnabled = enabled != 0;
        result = layer.displaySyncEnabled ? 2 : 1;
    }
    return result;
}

extern "C" const char *CaveWindowState() {
    static char description[512];
    for (NSWindow *window in NSApp.windows) {
        if (!FindViewLayer(window.contentView)) continue;
        NSRect bounds = window.contentView.bounds;
        snprintf(description, sizeof(description), "active=%d visible=%d style=%llu fullscreen=%d content=%.0fx%.0f scale=%.1f", (int)NSApp.active, (int)window.visible, (unsigned long long)window.styleMask, (int)((window.styleMask & NSWindowStyleMaskFullScreen) != 0), bounds.size.width, bounds.size.height, window.backingScaleFactor);
        return description;
    }
    return "No Metal window";
}

extern "C" int CaveWindowActive() { return NSApp.active ? 1 : 0; }
extern "C" int CaveWindowMatchesSize(int width, int height) {
    for (NSWindow *window in NSApp.windows) {
        if (!FindViewLayer(window.contentView)) continue;
        NSSize size = window.contentView.bounds.size;
        CGFloat scale = window.backingScaleFactor;
        return (int)lround(size.width * scale) == width && (int)lround(size.height * scale) == height;
    }
    return 0;
}
