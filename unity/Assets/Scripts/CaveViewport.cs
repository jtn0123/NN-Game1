using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Twenty native 32px tiles across, with integer scaling and a reserved HUD strip.
    public static class CaveViewport
    {
        public static bool Preview, HideHudStrip;
        public static int Scale
        {
            get
            {
                if (Preview) return 1;
                var fit = Mathf.Max(1, Mathf.Min(Screen.width / 640, Screen.height / (CaveSettings.Data.aspect == 1 ? 400 : 384)));
                return CaveSettings.Data.pixelScale == 0 ? fit : Mathf.Min(fit, CaveSettings.Data.pixelScale);
            }
        }
        public static float HudHeight => HideHudStrip ? 0 : CaveSettings.Data.hudScale == 2 ? 48 : 32;
        public static Rect Frame
        {
            get
            {
                var width = CaveSettings.Data.aspect == 2 ? Screen.width / Scale * Scale : Mathf.Min(Screen.width, 640 * Scale);
                var height = Mathf.Min((CaveSettings.Data.aspect == 1 ? 400 : 448) * Scale, Screen.height / Scale * Scale);
                return new Rect(Mathf.Floor((Screen.width - width) / 2), Mathf.Floor((Screen.height - height) / 2), width, height);
            }
        }
        public static float Width => Frame.width / Scale;
        public static float Height => Frame.height / Scale;
        public static float HudY => CaveVisualSettings.HudBottom ? Height - HudHeight : 0;
        public static float WorldTop => CaveVisualSettings.HudBottom ? 0 : HudHeight;
        public static float WorldBottom => CaveVisualSettings.HudBottom ? Height - HudHeight : Height;
        public static float HintY => CaveVisualSettings.HudBottom ? HudY - 28 : HudHeight + 4;
        public static Rect CameraRect
        {
            get
            {
                if (Preview) return new Rect(Mathf.Floor(Screen.width * .59f), Mathf.Floor(Screen.height * .34f), Mathf.Floor(Screen.width * .37f), Mathf.Floor(Screen.height * .39f));
                var frame = Frame;
                return new Rect(frame.x, Screen.height - frame.yMax + (CaveVisualSettings.HudBottom ? HudHeight * Scale : 0), frame.width, frame.height - HudHeight * Scale);
            }
        }
        public static Matrix4x4 Matrix => Matrix4x4.TRS(new Vector3(Frame.x, Frame.y, 0), Quaternion.identity, new Vector3(Scale, Scale, 1));
        public static Vector2 Point(Vector3 screen) => new Vector2((screen.x - Frame.x) / Scale, (Screen.height - screen.y - Frame.y) / Scale);
    }
}
