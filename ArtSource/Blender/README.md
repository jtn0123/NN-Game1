# Blender creature models

Every moving enemy (dinosaur, bat, snake, flying slime and walking rock) is
modelled and posed in Blender, then reduced to the game's palette pixel art.

```bash
python scripts/render_blender_sprites.py             # re-render every creature
python scripts/render_blender_sprites.py bat_enemy   # just one
```

- `creatures.py`: the models. Each part is authored in sprite-pixel
  coordinates with `at(col, row, depth)`, so features land on exact pixels.
- `sprite_ids.py`: the slot (palette ramp) and layer (behind, body or in front)
  encoded in the flat ID pass.
- `scripts/render_blender_sprites.py`: the palette ramps per creature and the
  conversion (4×4 majority sampling, shared brightness cuts, contour lines
  between layers, a one-pixel ink outline).
- `sprites/`: the committed results. `python -m src.unity_bridge.export_assets`
  reads these, so building or testing the game never needs Blender.

Blender 5.2 or newer is needed only to re-render (`brew install --cask blender`).
After re-rendering, run the creature art tests and re-record the baseline in
`tests/test_reference_slime_defeat_art.py` if the change was intentional.
