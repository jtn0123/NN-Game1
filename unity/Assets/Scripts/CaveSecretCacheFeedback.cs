using System;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    public static class CaveSecretCacheFeedback
    {
        public static string Marker(CaveEntity entity) => entity != null && entity.sprite == "secret_cache_armed"
            && !string.IsNullOrEmpty(entity.id) && entity.id.StartsWith("secret_cache_", StringComparison.Ordinal) ? "?" : null;
    }

    public sealed partial class CaveInterface
    {
        void DrawSecretCacheMarkers()
        {
            if (S.entities == null) return;
            foreach (var entity in S.entities)
            {
                var marker = CaveSecretCacheFeedback.Marker(entity);
                if (marker == null) continue;
                var point = CaveViewport.Point(game.GameCamera.WorldToScreenPoint(CaveWorld.Position(entity.x + entity.w / 2, entity.y - 7)));
                if (point.x < 6 || point.x > CaveViewport.Width - 12 || point.y < CaveViewport.WorldTop + 8 || point.y > CaveViewport.WorldBottom - 10) continue;
                Fill(new Rect(point.x - 4, point.y - 2, 8, 12), Color.black);
                PixelText(point.x - 3, point.y, marker, 1, paper);
            }
        }
    }
}
