using System.Collections.Generic;
using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Tile-aligned DOS-style room walls and industrial scenery, evenly lit.
    public sealed class CaveAtmosphere
    {
        readonly SpriteRenderer background, dressing;
        readonly List<SpriteRenderer> lights = new List<SpriteRenderer>();
        readonly Dictionary<SpriteRenderer, Vector3> anchors = new Dictionary<SpriteRenderer, Vector3>();
        readonly Sprite glow;
        readonly Material glowMaterial;
        readonly List<SpriteRenderer> accents = new List<SpriteRenderer>();
        readonly Transform parent;
        readonly Sprite pixel;
        readonly CaveCatalog catalog;
        readonly Dictionary<string, Sprite> sprites = new Dictionary<string, Sprite>();

        public CaveAtmosphere(Camera camera, Transform parent, Sprite glow, Material glowMaterial, Material stoneMaterial)
        {
            this.parent = parent; this.glow = glow; this.glowMaterial = glowMaterial;
            var texture = new Texture2D(1,1) { filterMode = FilterMode.Point }; texture.SetPixel(0,0,Color.white); texture.Apply();
            pixel = Sprite.Create(texture,new Rect(0,0,1,1),Vector2.one*.5f,32);
            background = Renderer("Patterned mine walls", parent, -20);
            dressing = Renderer("Service pipes, signs and crates", parent, -5);
            var asset = Resources.Load<TextAsset>("CaveCatalog");
            if (asset != null) catalog = JsonUtility.FromJson<CaveCatalog>(asset.text);
        }

        public int Theme(int level) => catalog != null && level >= 0 && level < catalog.caves.Length ? catalog.caves[level].theme : 0;

        public void Build(CaveSnapshot state)
        {
            foreach(var accent in accents) Object.Destroy(accent.gameObject); accents.Clear(); anchors.Clear();
            foreach(var light in lights) Object.Destroy(light.gameObject); lights.Clear();
            var center = new Vector3(state.cols / 2f, -state.rows / 2f, 0);
            var room = state.realm == "mine" ? "mine" : state.level.ToString();
            background.sprite = Load("wall_" + room);
            dressing.sprite = Load("dressing_" + room);
            background.transform.position = dressing.transform.position = center;
            // Ceiling drips are selected only over empty cells, never over a danger tile.
            for(var row=0;row<state.rows-3 && accents.Count<8;row++) for(var col=1;col<state.cols-1 && accents.Count<8;col++)
            {
                if((col*13+row*7+(state.level+2)*3)%41!=0 || state.layout[row][col]!='#' || state.layout[row+1][col]!='.')continue;
                var drip=Renderer("Optional ceiling drip",parent,-3); drip.sprite=pixel;
                drip.transform.position=CaveWorld.Position(col*32+16,row*32+34); drip.enabled=false; accents.Add(drip); anchors.Add(drip,drip.transform.localPosition);
                if(accents.Count>=8)break;
            }
            foreach(var entity in state.entities)
            {
                if(!entity.id.StartsWith("torch_"))continue;
                var light=Renderer("Optional torch light",parent,1); light.sprite=glow; light.sharedMaterial=glowMaterial;
                light.transform.position=CaveWorld.Position(entity.x+entity.w/2,entity.y+entity.h/2);
                light.transform.localScale=Vector3.one*2; light.enabled=false; lights.Add(light);
            }
            for(var i=0;i<3;i++)
            {
                var lamp=Renderer("Optional service lamp",parent,-4); lamp.sprite=pixel;
                // Attach to authored wall blocks so these cannot resemble pickups.
                for(var row=2;row<state.rows-2;row++)
                {
                    var col=2+(i*11+(state.level+2)*5)%Mathf.Max(1,state.cols-4);
                    if(state.layout[row][col]!='#')continue;
                    lamp.transform.position=CaveWorld.Position(col*32+5,row*32+5);break;
                }
                lamp.enabled=false; accents.Add(lamp);
            }
        }

        public void Animate(CaveSnapshot state)
        {
            foreach(var light in lights)
            { light.enabled=CaveSettings.Data.lighting>0; light.color=new Color(1,.67f,.33f,CaveSettings.Data.lighting/100f*.3f); }
            for(var i=0;i<accents.Count;i++)
            {
                var accent=accents[i];
                accent.enabled=CaveSettings.Data.environment && CaveVisualSettings.Motion && CaveSettings.Data.effectDensity>0;
                if(!accent.enabled)continue;
                if(accent.name=="Optional ceiling drip")
                {
                    var phase=Mathf.Repeat(CaveVisualClock.Now*.38f+i*.43f+Theme(state.level)*.17f,1);
                    accent.color=new Color(1f/3,1f/3,1,.7f);
                    accent.transform.localScale=new Vector3(1,phase<.6f?2:1,1);
                    // Move within the first empty tile below its ceiling anchor.
                    var p=anchors[accent]; p.y-=Mathf.Floor(phase*22)/32f;
                    accent.transform.localPosition=p;
                    accent.enabled=phase<.75f && (CaveSettings.Data.effectDensity==2||i%2==0);
                }
                else
                {
                    var on=(int)(CaveVisualClock.Now*.75f+i+Theme(state.level))%4!=0;
                    accent.color=on?new Color(1f/3,1,1f/3,.8f):new Color(0,1f/3,0,.8f);
                    accent.transform.localScale=new Vector3(2,1,1);
                }
            }
        }

        Sprite Load(string name)
        {
            if (sprites.TryGetValue(name, out var sprite)) return sprite;
            var texture = Resources.Load<Texture2D>("Environment/" + name);
            if (texture == null) { Debug.LogError("Missing cave environment: " + name); return null; }
            sprite = Sprite.Create(texture, new Rect(0, 0, texture.width, texture.height), Vector2.one * .5f, 32);
            sprites.Add(name, sprite); return sprite;
        }

        static SpriteRenderer Renderer(string name, Transform parent, int order)
        {
            var instance = new GameObject(name, typeof(SpriteRenderer)); instance.transform.SetParent(parent, false);
            var renderer = instance.GetComponent<SpriteRenderer>(); renderer.sortingOrder = order; return renderer;
        }
    }
}
