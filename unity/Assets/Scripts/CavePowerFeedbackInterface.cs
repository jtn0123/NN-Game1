using UnityEngine;

namespace CrystalCaves.Pilot
{
    public sealed partial class CaveInterface
    {
        void DrawPowerStatus()
        {
            if (CavePowerFeedback.Rows(S) == 0) return;
            var settings = CaveSettings.Data;
            var scale = CavePowerFeedback.TextScale(settings.hudScale);
            var bounds = CavePowerFeedback.Bounds(S, CaveViewport.Width, CaveViewport.WorldTop, CaveViewport.WorldBottom,
                settings.hudScale, settings.safeMargin);
            Fill(bounds, new Color(0, 0, 0, .96f));
            var y = bounds.y + 4;
            if (S.super_timer > 0)
            {
                var tint = settings.highContrast ? paper : gold;
                Fill(new Rect(bounds.x, y - 1, 2, 8 * scale + 2), tint);
                PixelText(bounds.x + 6, y, CavePowerFeedback.PowerText(S.super_timer), scale, tint);
                y += 8 * scale + 4;
            }
            if (S.freeze_timer > 0)
            {
                var tint = settings.highContrast ? paper : teal;
                Fill(new Rect(bounds.x, y - 1, 2, 8 * scale + 2), tint);
                PixelText(bounds.x + 6, y, CavePowerFeedback.FreezeText(S.freeze_timer), scale, tint);
            }
        }
    }
}
