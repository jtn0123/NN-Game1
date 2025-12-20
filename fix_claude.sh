#!/bin/bash
# Script to fix Claude Code extension/CLI issues

# Don't exit on errors - we want to continue through all fixes
set +e

echo "🔧 Claude Fix Script"
echo "==================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_error() {
    echo -e "${RED}❌ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

print_info() {
    echo -e "ℹ️  $1"
}

# Check if npm is available
check_npm() {
    if ! command -v npm &> /dev/null; then
        print_error "npm is not installed or not in PATH"
        print_info "Please install Node.js and npm first:"
        print_info "  brew install node  # macOS"
        print_info "  or visit: https://nodejs.org/"
        return 1
    fi
    print_success "npm is available: $(npm --version)"
    return 0
}

# Check if Claude CLI is installed
check_claude_cli() {
    if command -v claude &> /dev/null; then
        print_success "Claude CLI found: $(which claude)"
        claude --version 2>/dev/null || print_warning "Could not get Claude version"
        return 0
    else
        print_warning "Claude CLI not found in PATH"
        return 1
    fi
}

# Kill any running Claude processes
kill_claude_processes() {
    print_info "Checking for running Claude processes..."
    # Find and kill any claude processes
    pkill -f "claude" 2>/dev/null && print_info "Killed running Claude processes" || print_info "No running Claude processes found"
    # Wait a moment for processes to terminate
    sleep 1
}

# Force reinstall Claude Code CLI
force_reinstall_claude() {
    print_info "Force reinstalling @anthropic-ai/claude-code..."

    # Kill any running Claude processes that might lock files
    kill_claude_processes

    # Get the npm global directory
    NPM_GLOBAL=$(npm root -g)
    CLAUDE_MODULE_DIR="$NPM_GLOBAL/@anthropic-ai/claude-code"
    CLAUDE_BIN_DIR=$(npm config get prefix)/bin
    CLAUDE_BIN="$CLAUDE_BIN_DIR/claude"

    # First, try to uninstall if it exists
    print_info "Uninstalling existing installation..."
    UNINSTALL_OUTPUT=$(npm uninstall -g @anthropic-ai/claude-code 2>&1)
    UNINSTALL_EXIT=$?

    # If uninstall fails with ENOTEMPTY, manually remove the directory
    if [ $UNINSTALL_EXIT -ne 0 ] && echo "$UNINSTALL_OUTPUT" | grep -q "ENOTEMPTY"; then
        print_warning "npm uninstall failed with ENOTEMPTY error, manually removing directory..."

        # Remove the binary symlink first
        if [ -L "$CLAUDE_BIN" ] || [ -f "$CLAUDE_BIN" ]; then
            print_info "Removing Claude binary: $CLAUDE_BIN"
            rm -f "$CLAUDE_BIN" 2>/dev/null || sudo rm -f "$CLAUDE_BIN" 2>/dev/null || true
        fi

        # Remove the module directory
        if [ -d "$CLAUDE_MODULE_DIR" ]; then
            print_info "Removing module directory: $CLAUDE_MODULE_DIR"
            # Try without sudo first
            if rm -rf "$CLAUDE_MODULE_DIR" 2>/dev/null; then
                print_success "Removed module directory successfully"
            else
                print_warning "Need sudo to remove directory, you may be prompted for password..."
                if sudo rm -rf "$CLAUDE_MODULE_DIR" 2>/dev/null; then
                    print_success "Removed module directory successfully (with sudo)"
                else
                    print_error "Failed to remove directory even with sudo"
                    print_info "You may need to manually remove: $CLAUDE_MODULE_DIR"
                    return 1
                fi
            fi
        fi

        # Wait a moment for filesystem to sync
        sleep 1
    elif [ $UNINSTALL_EXIT -eq 0 ]; then
        print_success "Successfully uninstalled via npm"
    fi

    # Clear npm cache for this package
    print_info "Clearing npm cache..."
    npm cache clean --force 2>&1 | head -5 > /dev/null

    # Check npm permissions
    print_info "Checking npm permissions..."
    NPM_PREFIX=$(npm config get prefix)
    if [ ! -w "$NPM_PREFIX" ]; then
        print_warning "npm prefix directory may not be writable: $NPM_PREFIX"
        print_info "You may need to run with sudo or fix permissions"
    fi

    # Install with force flag
    print_info "Installing @anthropic-ai/claude-code (this may take a moment)..."
    if npm install -g @anthropic-ai/claude-code --force 2>&1; then
        print_success "Claude Code CLI installed successfully"
        # Return 0 to indicate fresh install (skip manual update)
        return 0
    else
        print_warning "Regular install failed, trying with sudo..."
        print_info "You may be prompted for your password..."
        if sudo npm install -g @anthropic-ai/claude-code --force 2>&1; then
            print_success "Claude Code CLI installed successfully (with sudo)"
            return 0
        else
            print_error "Failed to install even with sudo"
            print_info "Manual fix required - try:"
            print_info "  sudo rm -rf $CLAUDE_MODULE_DIR"
            print_info "  sudo npm install -g @anthropic-ai/claude-code --force"
            return 1
        fi
    fi
}

# Install/reinstall Claude Code CLI
install_claude_cli() {
    # Use force_reinstall_claude which handles ENOTEMPTY errors and other edge cases
    force_reinstall_claude
}

# Run Claude doctor if available
run_claude_doctor() {
    if command -v claude &> /dev/null; then
        # Skip doctor command - it requires interactive terminal and raw mode
        # which doesn't work well in script contexts. Users can run it manually.
        print_info "Skipping 'claude doctor' - requires interactive terminal"
        print_info "To run diagnostics manually, use: claude doctor"
        return 0
    else
        print_warning "Claude CLI not available, skipping doctor command"
        return 1
    fi
}

# Try to manually update Claude
manual_update_claude() {
    if command -v claude &> /dev/null; then
        print_info "Attempting manual update..."
        # Try to trigger update via npm
        npm update -g @anthropic-ai/claude-code 2>&1 || true
        print_info "Manual update attempt completed"
    fi
}

# Fix auto-update permissions
fix_auto_update() {
    print_info "Checking auto-update configuration..."

    if command -v claude &> /dev/null; then
        # Check if we can write to the installation directory
        CLAUDE_PATH=$(which claude)
        CLAUDE_DIR=$(dirname "$CLAUDE_PATH")

        if [ -w "$CLAUDE_DIR" ]; then
            print_success "Claude installation directory is writable: $CLAUDE_DIR"
        else
            print_warning "Claude installation directory may not be writable: $CLAUDE_DIR"
            print_info "This could cause auto-update failures"
            print_info "You may need to fix permissions or use sudo for updates"
        fi

        # Check npm global directory permissions
        NPM_GLOBAL=$(npm root -g)
        if [ -w "$NPM_GLOBAL" ]; then
            print_success "npm global directory is writable: $NPM_GLOBAL"
        else
            print_warning "npm global directory may not be writable: $NPM_GLOBAL"
        fi
    fi
}

# Fix zsh configuration issues
fix_zsh_config() {
    print_info "Checking zsh configuration..."

    ZSHRC="$HOME/.zshrc"
    if [ ! -f "$ZSHRC" ]; then
        print_warning ".zshrc not found, creating basic one..."
        touch "$ZSHRC"
    fi

    # Check for common syntax issues
    if zsh -n "$ZSHRC" 2>&1; then
        print_success "zsh configuration syntax is valid"
    else
        print_error "zsh configuration has syntax errors!"
        print_info "Run 'zsh -n ~/.zshrc' to see details"
        print_info "Common issues:"
        print_info "  - Unmatched parentheses ()"
        print_info "  - Unmatched braces {}"
        print_info "  - Unmatched brackets []"
        return 1
    fi

    return 0
}

# Main execution
main() {
    echo ""
    print_info "Step 1: Checking npm..."
    if ! check_npm; then
        exit 1
    fi

    echo ""
    print_info "Step 2: Checking zsh configuration..."
    fix_zsh_config

    echo ""
    print_info "Step 3: Checking Claude CLI..."
    CLAUDE_INSTALLED=false
    FRESH_INSTALL=false
    if check_claude_cli; then
        CLAUDE_INSTALLED=true
        echo ""
        print_info "Step 4: Fixing auto-update permissions..."
        fix_auto_update
        echo ""
        print_info "Step 5: Force reinstalling Claude CLI to fix auto-update..."
        if force_reinstall_claude; then
            FRESH_INSTALL=true
        fi
        echo ""
        # Only attempt manual update if we didn't just do a fresh install
        if [ "$FRESH_INSTALL" = false ]; then
            print_info "Step 6: Attempting manual update..."
            manual_update_claude
        else
            print_info "Step 6: Skipping manual update (fresh install completed)"
        fi
    else
        echo ""
        print_info "Step 4: Installing Claude CLI..."
        if install_claude_cli; then
            CLAUDE_INSTALLED=true
            FRESH_INSTALL=true
        fi
    fi

    if [ "$CLAUDE_INSTALLED" = true ]; then
        echo ""
        print_info "Step 7: Running Claude doctor..."
        run_claude_doctor
    fi

    echo ""
    echo "==================="
    print_success "Fix script completed!"
    echo ""
    print_info "Next steps:"
    print_info "1. Restart your terminal or run: source ~/.zshrc"
    print_info "2. Verify Claude CLI works: claude --version"
    print_info "3. The auto-update should now work. If not, try:"
    print_info "   npm install -g @anthropic-ai/claude-code --force"
    print_info "4. If issues persist, check: claude doctor"
}

# Run main function
main
