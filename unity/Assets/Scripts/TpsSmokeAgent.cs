using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace TpsTransfer
{
    [RequireComponent(typeof(Rigidbody))]
    public sealed class TpsSmokeAgent : Agent
    {
        public Transform YawRig;
        public Transform PitchRig;

        private Rigidbody body;
        private Vector4 heldAction;
        private Vector3 horizontalVelocity;
        private int physicsTicks;
        private float yawDegrees;
        private float pitchDegrees;

        public override void Initialize()
        {
            body = GetComponent<Rigidbody>();
            Time.fixedDeltaTime = 0.02f;
            MaxStep = 0; // The experiment counts 10 Hz decisions explicitly.
        }

        public override void OnEpisodeBegin()
        {
            body.position = new Vector3(0f, 1f, 0f);
            body.rotation = Quaternion.identity;
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            horizontalVelocity = Vector3.zero;
            heldAction = Vector4.zero;
            physicsTicks = 0;
            yawDegrees = 0f;
            pitchDegrees = 15f;
            UpdateCameraRig();
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // RGB comes only from CameraSensorComponent. No privileged state.
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            var values = actions.ContinuousActions;
            heldAction = new Vector4(
                Mathf.Clamp(values[0], -1f, 1f),
                Mathf.Clamp(values[1], -1f, 1f),
                Mathf.Clamp(values[2], -1f, 1f),
                Mathf.Clamp(values[3], -1f, 1f));
            AddReward(-0.001f);
        }

        private void FixedUpdate()
        {
            if (physicsTicks % 5 == 0)
            {
                RequestDecision();
            }

            var delta = Time.fixedDeltaTime;
            yawDegrees += heldAction.z * 120f * delta;
            pitchDegrees = Mathf.Clamp(pitchDegrees + heldAction.w * 90f * delta, -60f, 60f);
            UpdateCameraRig();

            var input = new Vector2(heldAction.x, heldAction.y);
            if (input.sqrMagnitude > 1f)
            {
                input.Normalize();
            }

            var forward = YawRig.forward;
            var right = YawRig.right;
            forward.y = 0f;
            right.y = 0f;
            var desired = (right.normalized * input.x + forward.normalized * input.y) * 3f;
            horizontalVelocity = Vector3.MoveTowards(horizontalVelocity, desired, 6f * delta);
            body.MovePosition(body.position + horizontalVelocity * delta);

            physicsTicks++;
            if (physicsTicks >= 500)
            {
                EpisodeInterrupted();
            }
        }

        private void UpdateCameraRig()
        {
            YawRig.localRotation = Quaternion.Euler(0f, yawDegrees, 0f);
            PitchRig.localRotation = Quaternion.Euler(pitchDegrees, 0f, 0f);
        }
    }
}
