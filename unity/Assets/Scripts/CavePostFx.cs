using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Image effects run on the world camera before IMGUI draws the crisp HUD.
    [RequireComponent(typeof(Camera))]
    public sealed class CavePostFx : MonoBehaviour
    {
        const int Composite = 0, BrightPass = 1, Blur = 2;
        Material material;
        public static bool Needed
        {
            get
            {
                var d = CaveSettings.Data;
                return d.brightness != 100 || d.palette != 0 || d.scanlines > 0 || d.phosphor > 0 || d.bloom > 0 || d.vignette > 0;
            }
        }
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
            material.SetFloat("_Vignette", d.vignette / 100f);
            material.SetFloat("_Aspect", source.width / (float)Mathf.Max(1, source.height));
            RenderTexture near = null, wide = null;
            if (d.bloom > 0)
            {
                // Two blur radii: a tight halo around crystals and shots, plus a wide soft glow.
                material.SetFloat("_BloomThreshold", .58f);
                near = Blurred(source, Mathf.Max(1, source.width / 4), Mathf.Max(1, source.height / 4), true);
                wide = Blurred(near, Mathf.Max(1, source.width / 16), Mathf.Max(1, source.height / 16), false);
                material.SetTexture("_Bloom", near);
                material.SetTexture("_BloomWide", wide);
            }
            material.SetFloat("_BloomIntensity", d.bloom > 0 ? d.bloom / 50f : 0);
            Graphics.Blit(source, destination, material, Composite);
            if (near != null) RenderTexture.ReleaseTemporary(near);
            if (wide != null) RenderTexture.ReleaseTemporary(wide);
        }

        // The first level extracts bright light; deeper levels blur what is already extracted.
        RenderTexture Blurred(RenderTexture input, int width, int height, bool extract)
        {
            var target = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.DefaultHDR);
            var scratch = RenderTexture.GetTemporary(width, height, 0, RenderTextureFormat.DefaultHDR);
            target.filterMode = scratch.filterMode = FilterMode.Bilinear;
            if (extract) Graphics.Blit(input, target, material, BrightPass);
            else Graphics.Blit(input, target);
            for (var pass = 0; pass < 2; pass++)
            {
                material.SetVector("_Direction", new Vector2(1, 0)); Graphics.Blit(target, scratch, material, Blur);
                material.SetVector("_Direction", new Vector2(0, 1)); Graphics.Blit(scratch, target, material, Blur);
            }
            RenderTexture.ReleaseTemporary(scratch);
            return target;
        }
        void OnDestroy() { if (material != null) Destroy(material); }
    }
}
