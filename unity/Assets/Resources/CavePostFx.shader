Shader "CrystalCaves/World presentation"
{
    Properties { _MainTex ("World", 2D) = "white" {} }
    SubShader
    {
        Cull Off ZWrite Off ZTest Always
        Pass
        {
            CGPROGRAM
            #pragma vertex vert_img
            #pragma fragment frag
            #include "UnityCG.cginc"
            sampler2D _MainTex;
            float _Brightness, _Palette, _Scanlines, _Phosphor, _Rows, _Columns;
            fixed4 frag(v2f_img i) : SV_Target
            {
                float3 c = tex2D(_MainTex, i.uv).rgb;
                if (_Palette > 1.5) c *= float3(.94, 1, 1.08);
                else if (_Palette > .5) c *= float3(1.08, 1, .92);
                c *= _Brightness;
                c *= 1 - _Scanlines * fmod(floor(i.uv.y * _Rows), 2);
                float column = fmod(floor(i.uv.x * _Columns), 3);
                float3 mask = column < 1 ? float3(1,.82,.82) : column < 2 ? float3(.82,1,.82) : float3(.82,.82,1);
                c *= lerp(float3(1,1,1), mask, _Phosphor);
                return fixed4(saturate(c), 1);
            }
            ENDCG
        }
    }
    Fallback Off
}
