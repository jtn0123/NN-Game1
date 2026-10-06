using UnityEngine;

namespace CrystalCaves.Pilot
{
    // Shared by mouse, keyboard and the legacy joystick axes already shipped with the game.
    public sealed class CaveMenuNavigation
    {
        public int Focus { get; private set; }
        public int TabDelta { get; private set; }
        public int Adjust { get; private set; }
        public bool Submit { get; private set; }
        public bool Back { get; private set; }
        public bool Controller { get; private set; }
        float repeatAt;
        int previousDirection;
        public void Reset() { Focus = 0; previousDirection = 0; repeatAt = 0; }
        public void Poll(int count)
        {
            var vertical = Input.GetAxisRaw("Vertical");
            var horizontal = Input.GetAxisRaw("Horizontal");
            var direction = Mathf.Abs(vertical) > .55f ? vertical > 0 ? -1 : 1 : 0;
            var adjust = Mathf.Abs(horizontal) > .55f ? horizontal > 0 ? 1 : -1 : 0;
            var anyJoystick = Input.GetKeyDown(KeyCode.JoystickButton0) || Input.GetKeyDown(KeyCode.JoystickButton1)
                || Input.GetKeyDown(KeyCode.JoystickButton4) || Input.GetKeyDown(KeyCode.JoystickButton5);
            if (anyJoystick || (!Input.anyKey && (direction != 0 || adjust != 0))) Controller = true;
            if (Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.Tab) || Input.GetKeyDown(KeyCode.UpArrow)
                || Input.GetKeyDown(KeyCode.DownArrow) || Input.GetKeyDown(KeyCode.LeftArrow) || Input.GetKeyDown(KeyCode.RightArrow)) Controller = false;
            TabDelta = Input.GetKeyDown(KeyCode.Tab) ? (Input.GetKey(KeyCode.LeftShift) || Input.GetKey(KeyCode.RightShift) ? -1 : 1)
                : Input.GetKeyDown(KeyCode.JoystickButton4) ? -1 : Input.GetKeyDown(KeyCode.JoystickButton5) ? 1 : 0;
            var repeat = direction != 0 ? direction : adjust * 2;
            var pulse = repeat != 0 && (repeat != previousDirection || Time.unscaledTime >= repeatAt);
            if (pulse) repeatAt = Time.unscaledTime + (repeat != previousDirection ? .35f : .14f);
            previousDirection = repeat;
            Apply(count, pulse ? direction : 0, pulse ? adjust : 0,
                Input.GetKeyDown(KeyCode.Return) || Input.GetKeyDown(KeyCode.KeypadEnter) || Input.GetKeyDown(KeyCode.JoystickButton0),
                Input.GetKeyDown(KeyCode.Escape) || Input.GetKeyDown(KeyCode.P) || Input.GetKeyDown(KeyCode.JoystickButton1), TabDelta);
        }
        public void Apply(int count, int move, int adjust, bool submit, bool back, int tab)
        {
            Focus = count < 1 ? 0 : ((Focus + move) % count + count) % count;
            Adjust = adjust; Submit = submit; Back = back; TabDelta = tab;
        }
        public void PointAt(int index) { Focus = index; }
        public bool Activate(int index) { if (Focus != index || !Submit) return false; Submit = false; return true; }
        public int Adjustment(int index) { if (Focus != index) return 0; var result = Adjust; Adjust = 0; return result; }
        public void Clear() { Submit = Back = false; Adjust = TabDelta = 0; }
    }
}
