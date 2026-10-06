using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Image effects run on the world camera before IMGUI draws the crisp HUD.
    [RequireComponent(typeof(Camera))]
    public sealed class CavePostFx : MonoBehaviour
    {
        Material material;
        public static bool Needed => CaveSettings.Data.brightness != 100 || CaveSettings.Data.palette != 0 || CaveSettings.Data.scanlines > 0 || CaveSettings.Data.phosphor > 0;
        void Awake() { material = new Material(Resources.Load<Shader>("CavePostFx")); }
        void OnRenderImage(RenderTexture source, RenderTexture destination)
        {
            source.filterMode = FilterMode.Point;
            var d = CaveSettings.Data;
            material.SetFloat("_Brightness", d.brightness / 100f);
            material.SetFloat("_Palette", d.palette);
            material.SetFloat("_Scanlines", d.scanlines / 100f);
            material.SetFloat("_Phosphor", d.phosphor / 100f);
            material.SetFloat("_Rows", source.height / (float)CaveViewport.Scale);
            material.SetFloat("_Columns", source.width);
            Graphics.Blit(source, destination, material);
        }
        void OnDestroy() { if (material != null) Destroy(material); }
    }
}
