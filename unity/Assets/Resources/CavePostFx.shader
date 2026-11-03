Shader "CrystalCaves/World presentation"
{
    Properties { _MainTex ("World", 2D) = "white" {} _Bloom ("Bloom", 2D) = "black" {} _BloomWide ("Wide bloom", 2D) = "black" {} }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        CGINCLUDE
        #include "UnityCG.cginc"
        sampler2D _MainTex, _Bloom, _BloomWide;
        float4 _MainTex_TexelSize;
        float _Brightness, _Palette, _Scanlines, _Phosphor, _Rows, _Columns;
        float _BloomIntensity, _BloomThreshold, _Vignette, _Aspect;
        float2 _Direction;
        ENDCG

        // 0: composite. Bloom is added before grading so palette and brightness apply to light too.
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            fixed4 frag(v2f_img i) : SV_Target
            {
                float3 c = tex2D(_MainTex, i.uv).rgb;
                if (_BloomIntensity > 0)
                    c += (tex2D(_Bloom, i.uv).rgb * .65 + tex2D(_BloomWide, i.uv).rgb * .85) * _BloomIntensity;
                if (_Palette > 1.5) c *= float3(.94, 1, 1.08);
                else if (_Palette > .5) c *= float3(1.08, 1, .92);
                c *= _Brightness;
                if (_Vignette > 0)
                {
                    float2 offset = (i.uv - .5) * float2(_Aspect, 1);
                    c *= 1 - _Vignette * smoothstep(.32, .95, length(offset) * 1.15);
                }
                c *= 1 - _Scanlines * fmod(floor(i.uv.y * _Rows), 2);
                float column = fmod(floor(i.uv.x * _Columns), 3);
                float3 mask = column < 1 ? float3(1,.82,.82) : column < 2 ? float3(.82,1,.82) : float3(.82,.82,1);
                c *= lerp(float3(1,1,1), mask, _Phosphor);
                return fixed4(saturate(c), 1);
            }
            ENDCG
        }

        // 1: bright pass with a soft knee, averaging four source texels while downsampling.
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            float3 Bright(float2 uv)
            {
                float3 c = tex2D(_MainTex, uv).rgb;
                float peak = max(c.r, max(c.g, c.b));
                float knee = saturate((peak - _BloomThreshold) / max(1e-4, 1 - _BloomThreshold));
                return c * knee * knee;
            }
            fixed4 frag(v2f_img i) : SV_Target
            {
                float2 t = _MainTex_TexelSize.xy;
                float3 c = Bright(i.uv + t * float2(-.5, -.5)) + Bright(i.uv + t * float2(.5, -.5))
                         + Bright(i.uv + t * float2(-.5, .5)) + Bright(i.uv + t * float2(.5, .5));
                return fixed4(c * .25, 1);
            }
            ENDCG
        }

        // 2: separable 9-tap Gaussian along _Direction (in texels).
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            fixed4 frag(v2f_img i) : SV_Target
            {
                float2 step = _MainTex_TexelSize.xy * _Direction;
                float3 c = tex2D(_MainTex, i.uv).rgb * .2270;
                c += (tex2D(_MainTex, i.uv + step * 1.3846).rgb + tex2D(_MainTex, i.uv - step * 1.3846).rgb) * .3162;
                c += (tex2D(_MainTex, i.uv + step * 3.2308).rgb + tex2D(_MainTex, i.uv - step * 3.2308).rgb) * .0703;
                return fixed4(c, 1);
            }
            ENDCG
        }
    }
    Fallback Off
}
