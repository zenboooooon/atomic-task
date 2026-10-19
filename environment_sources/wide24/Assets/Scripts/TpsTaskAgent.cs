using System;
using System.Collections.Generic;
using System.IO;
using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

namespace TpsTransfer
{
    [Serializable]
    public sealed class TpsObstacleRecord
    {
        public float x;
        public float z;
        public float radius;
        public float height;
    }

    [Serializable]
    public sealed class TpsLayoutRecord
    {
        public string layout_id;
        public string task;
        public string split;
        public float player_x;
        public float player_z;
        public float target_x;
        public float target_z;
        public float camera_yaw_degrees;
        public float look_start_viewport_x;
        public float initial_speed_mps;
        public TpsObstacleRecord[] obstacles;
        public bool has_platform;
        public float platform_x;
        public float platform_z;
        public float platform_yaw_degrees;
        public float target_y;
    }

    [Serializable]
    public sealed class TpsLayoutManifest
    {
        public string protocol_version;
        public TpsLayoutRecord[] layouts;
    }

    [RequireComponent(typeof(Rigidbody))]
    public sealed class TpsTaskAgent : Agent
    {
        private const string RewardVersion = "gaze-jump-v0.10-single-color";
        public Transform YawRig;
        public Transform PitchRig;
        public Camera PolicyCamera;
        public Transform Target;
        public GameObject[] Obstacles;
        public string TaskOverride;
        public string HeadId;
        public string ColorVariant = "none";
        public BoxCollider Platform;
        public Vector3 ArenaOrigin;
        public int LayoutStartOffset;
        public int LayoutStride = 1;

        private Rigidbody body;
        private Vector4 heldAction;
        private Vector3 horizontalVelocity;
        private Vector3 startPosition;
        private List<TpsLayoutRecord> layouts;
        private TpsLayoutRecord layout;
        private string task;
        private string split;
        private int nextLayout;
        private int physicsTicks;
        private int holdTicks;
        private int visibleTicks;
        private bool firstStopEntry;
        private bool alignedOnce;
        private bool bandEntered;
        private bool treasureFound;
        private int treasureFirstSeenTick;
        private float firstEntryVisibleFraction;
        private float yawDegrees;
        private float pitchDegrees;
        private float previousCosine;
        private float previousDistance;
        private float previousBandError;
        private bool episodeEnding;
        private bool actionStarted;
        private bool jumpHeld;
        private bool jumpPending;
        private bool airborneAfterJump;
        private bool landedOnPlatform;
        private int jumpCount;
        private string scriptedProbe;
        private int probePhase;
        private bool movingObstacles;
        private Vector3[] obstacleBases;
        private float rewardTime;
        private float rewardView;
        private float rewardDistance;
        private float rewardBand;
        private float penaltyLookMovement;
        private float penaltyStopSpeed;
        private float penaltyInvisible;
        private float rewardTerminal;
        private float rewardMilestone;
        private float penaltyBandDistance;
        private float penaltyLookCommand;
        private float penaltyStopView;
        private float penaltyStopCameraCommand;
        private float penaltyCameraCommand;

        public override void Initialize()
        {
            Time.fixedDeltaTime = 0.02f;
            MaxStep = 0;
            body = GetComponent<Rigidbody>();
            scriptedProbe = ArgumentValue("--tps-scripted-probe", "");
            // Agent observations use the current physics pose. Interpolated
            // render poses can blend the previous layout into a reset frame.
            body.interpolation = RigidbodyInterpolation.None;
            // The RGB sensor renders a square texture. Camera.aspect set in an
            // editor builder is not serialized, so fix it on the runtime camera
            // before using WorldToViewportPoint for starts and visibility.
            PolicyCamera.aspect = 1f;
            string requestedTask = ArgumentValue("--tps-task", "look");
            task = requestedTask == "pretrain" ? TaskOverride : requestedTask;
            if (string.IsNullOrEmpty(task))
            {
                throw new ArgumentException("Multi-task player needs a TaskOverride");
            }
            split = ArgumentValue("--tps-split", "train");
            string obstacleMode = ArgumentValue("--tps-obstacle-motion", "static");
            if (obstacleMode != "static" && obstacleMode != "moving")
            {
                throw new ArgumentException("--tps-obstacle-motion must be static or moving");
            }
            movingObstacles = task == "treasure" && obstacleMode == "moving";
            obstacleBases = new Vector3[Obstacles.Length];
            int startIndex;
            if (!int.TryParse(ArgumentValue("--tps-layout-start", "0"), out startIndex) || startIndex < 0)
            {
                throw new ArgumentException("--tps-layout-start must be nonnegative");
            }
            if (LayoutStride < 1)
            {
                throw new ArgumentException("LayoutStride must be positive");
            }
            nextLayout = startIndex + LayoutStartOffset;
            string path = Path.Combine(Application.streamingAssetsPath, "pilot_layouts.json");
            var manifest = JsonUtility.FromJson<TpsLayoutManifest>(File.ReadAllText(path));
            if (manifest == null || manifest.layouts == null
                || (manifest.protocol_version != "pilot-v0.1"
                    && manifest.protocol_version != "pilot-v0.2"
                    && manifest.protocol_version != "treasure-v0.1"
                    && manifest.protocol_version != "treasure-v0.2"
                    && manifest.protocol_version != "primitive-contact-v0.3"
                    && manifest.protocol_version != "gaze-v0.4"
                    && manifest.protocol_version != "gaze-v0.5-look1p5"
                    && manifest.protocol_version != "gaze-v0.6-edge-appearance"
                    && manifest.protocol_version != "gaze-v0.7-free-look0p5"
                    && manifest.protocol_version != "gaze-v0.8-color-variants"
                    && manifest.protocol_version != "gaze-jump-v0.10-single-color"))
            {
                throw new InvalidDataException("Invalid pilot layout manifest");
            }
            layouts = new List<TpsLayoutRecord>();
            foreach (var candidate in manifest.layouts)
            {
                if (candidate.task == task && candidate.split == split)
                {
                    layouts.Add(candidate);
                }
            }
            if (layouts.Count == 0)
            {
                throw new InvalidDataException("No layouts for task=" + task + ", split=" + split);
            }
        }

        public override void OnEpisodeBegin()
        {
            // Relocate colliders while inactive. A contact from the previous
            // layout must not terminate the new episode before its first action.
            episodeEnding = true;
            actionStarted = false;
            jumpHeld = false;
            jumpPending = false;
            airborneAfterJump = false;
            landedOnPlatform = false;
            jumpCount = 0;
            probePhase = 0;
            rewardTime = 0f;
            rewardView = 0f;
            rewardDistance = 0f;
            rewardBand = 0f;
            penaltyLookMovement = 0f;
            penaltyStopSpeed = 0f;
            penaltyInvisible = 0f;
            rewardTerminal = 0f;
            rewardMilestone = 0f;
            penaltyBandDistance = 0f;
            penaltyLookCommand = 0f;
            penaltyStopView = 0f;
            penaltyStopCameraCommand = 0f;
            penaltyCameraCommand = 0f;
            layout = layouts[nextLayout % layouts.Count];
            nextLayout += LayoutStride;
            Vector3 resetPosition = ArenaOrigin + new Vector3(layout.player_x, 1f, layout.player_z);
            transform.SetPositionAndRotation(resetPosition, Quaternion.identity);
            body.position = resetPosition;
            body.rotation = Quaternion.identity;
            body.linearVelocity = Vector3.zero;
            body.angularVelocity = Vector3.zero;
            startPosition = body.position;
            yawDegrees = layout.camera_yaw_degrees;
            pitchDegrees = 10f;
            heldAction = Vector4.zero;
            horizontalVelocity = Vector3.zero;
            physicsTicks = 0;
            holdTicks = 0;
            visibleTicks = 0;
            firstStopEntry = false;
            alignedOnce = false;
            bandEntered = false;
            treasureFound = false;
            treasureFirstSeenTick = -1;
            firstEntryVisibleFraction = 0f;
            Target.position = ArenaOrigin + new Vector3(layout.target_x,
                layout.target_y > 0f ? layout.target_y : 1.6f, layout.target_z);
            if (Platform != null)
            {
                Platform.gameObject.SetActive(false);
                if (layout.has_platform)
                {
                    Platform.transform.SetPositionAndRotation(ArenaOrigin +
                        new Vector3(layout.platform_x, 0.5f, layout.platform_z),
                        Quaternion.Euler(0f, layout.platform_yaw_degrees, 0f));
                    Platform.gameObject.SetActive(true);
                }
            }
            for (int index = 0; index < Obstacles.Length; index++)
            {
                bool active = layout.obstacles != null && index < layout.obstacles.Length;
                Obstacles[index].SetActive(false);
                if (active)
                {
                    var obstacle = layout.obstacles[index];
                    Obstacles[index].transform.position = ArenaOrigin + new Vector3(obstacle.x, obstacle.height / 2f, obstacle.z);
                    obstacleBases[index] = Obstacles[index].transform.position;
                    Obstacles[index].transform.localScale =
                        new Vector3(obstacle.radius * 2f, obstacle.height / 2f, obstacle.radius * 2f);
                    Obstacles[index].SetActive(true);
                }
            }
            if (task == "stop")
            {
                Vector3 toward = Target.position - body.position;
                toward.y = 0f;
                horizontalVelocity = toward.normalized * layout.initial_speed_mps;
                body.linearVelocity = horizontalVelocity;
            }
            UpdateCameraRig();
            Physics.SyncTransforms();
            if (task == "look" && layout.look_start_viewport_x > 0f)
            {
                // Solve against the actual camera projection, including the
                // shoulder offset and parent transforms, before the first RGB.
                Vector3 toward = Target.position - body.position;
                float bearing = Mathf.Atan2(toward.x, toward.z) * Mathf.Rad2Deg;
                float lower = bearing - 85f;
                float upper = bearing + 85f;
                for (int iteration = 0; iteration < 40; iteration++)
                {
                    yawDegrees = (lower + upper) * 0.5f;
                    UpdateCameraRig();
                    float x = PolicyCamera.WorldToViewportPoint(Target.position).x;
                    if (x > layout.look_start_viewport_x) lower = yawDegrees;
                    else upper = yawDegrees;
                }
                yawDegrees = (lower + upper) * 0.5f;
                UpdateCameraRig();
                Vector3 initialViewport = PolicyCamera.WorldToViewportPoint(Target.position);
                Debug.Log("TPS_LOOK_START layout=" + layout.layout_id
                    + " head=" + HeadId + " color_variant=" + ColorVariant
                    + " viewport_x=" + initialViewport.x + " viewport_y=" + initialViewport.y
                    + " aspect=" + PolicyCamera.aspect);
            }
            previousDistance = TargetDistance();
            previousBandError = BandError(previousDistance);
            previousCosine = ViewCosineForReward();
            Physics.SyncTransforms();
            episodeEnding = false;
        }

        public override void CollectObservations(VectorSensor sensor)
        {
            // Policy and value model receive only the CameraSensorComponent RGB.
        }

        public override void OnActionReceived(ActionBuffers actions)
        {
            if (!actionStarted)
            {
                actionStarted = true;
                previousDistance = TargetDistance();
                previousBandError = BandError(previousDistance);
                previousCosine = ViewCosineForReward();
            }
            var values = actions.ContinuousActions;
            heldAction = new Vector4(
                Mathf.Clamp(values[0], -1f, 1f),
                Mathf.Clamp(values[1], -1f, 1f),
                Mathf.Clamp(values[2], -1f, 1f),
                Mathf.Clamp(values[3], -1f, 1f));
            bool pressed = values[4] > 0.5f;
            if (pressed && !jumpHeld) jumpPending = true;
            jumpHeld = pressed;
            AddComponent(ref rewardTime, -0.001f);
            if (task == "stop")
            {
                AddComponent(ref penaltyBandDistance, -0.01f * BandError(TargetDistance()));
                AddComponent(ref penaltyStopView, -0.01f * (1f - ViewCosine()));
                AddComponent(ref penaltyStopCameraCommand,
                    -0.01f * (Mathf.Abs(heldAction.z) + Mathf.Abs(heldAction.w)));
            }
            if (task == "look" || task == "approach" || task == "jump" || task == "treasure")
            {
                AddComponent(ref penaltyCameraCommand,
                    -0.001f * (Mathf.Abs(heldAction.z) + Mathf.Abs(heldAction.w)));
            }
        }

        private void FixedUpdate()
        {
            if (!string.IsNullOrEmpty(scriptedProbe) && Time.timeSinceLevelLoad > 30f)
            {
                Application.Quit(0);
                return;
            }
            if (layout == null || episodeEnding)
            {
                return;
            }
            float delta = Time.fixedDeltaTime;
            yawDegrees += heldAction.z * 90f * delta;
            pitchDegrees = Mathf.Clamp(pitchDegrees + heldAction.w * 90f * delta, -60f, 60f);
            UpdateCameraRig();
            if (movingObstacles)
            {
                MoveObstacles();
            }
            Vector2 input = new Vector2(heldAction.x, heldAction.y);
            if (input.sqrMagnitude > 1f)
            {
                input.Normalize();
            }
            Vector3 forward = YawRig.forward;
            Vector3 right = YawRig.right;
            forward.y = 0f;
            right.y = 0f;
            Vector3 desired = (right.normalized * input.x + forward.normalized * input.y) * 3f;
            horizontalVelocity = Vector3.MoveTowards(horizontalVelocity, desired, 6f * delta);
            // Five-action control: preserve vertical gravity/collision dynamics.
            Vector3 velocity = body.linearVelocity;
            velocity.x = horizontalVelocity.x;
            velocity.z = horizontalVelocity.z;
            if (jumpPending)
            {
                if (Physics.Raycast(body.position, Vector3.down, out var ground, 1.06f)
                    && ground.normal.y > 0.7f && Mathf.Abs(velocity.y) < 0.3f)
                {
                    velocity.y = 6f;
                    jumpCount++;
                }
                jumpPending = false;
            }
            body.linearVelocity = velocity;
            if (jumpCount > 0 && body.position.y - ArenaOrigin.y > 1.1f)
                airborneAfterJump = true;
            physicsTicks++;

            bool visible = TargetVisible();
            if (visible)
            {
                visibleTicks++;
            }
            // A reset can run FixedUpdate before the first policy action.
            // Discovery reward belongs to that action, not the new observation.
            if (task == "treasure" && actionStarted && visible && !treasureFound)
            {
                treasureFound = true;
                treasureFirstSeenTick = physicsTicks;
                AddComponent(ref rewardMilestone, 0.5f);
            }
            float distance = TargetDistance();
            float cosine = ViewCosineForReward();
            float bandError = BandError(distance);
            if (actionStarted)
            {
                if (task == "look" || task == "composite")
                {
                    AddComponent(ref rewardView, 5f * (cosine - previousCosine));
                }
                if (task == "treasure" && visible && ViewAngle() <= 7f)
                {
                    AddComponent(ref rewardView, 0.0003f);
                }
                if (task == "approach" || task == "avoid" || task == "composite"
                    || (task == "treasure" && treasureFound))
                {
                    AddComponent(ref rewardDistance, 0.1f * (previousDistance - distance));
                }
                if (task == "stop")
                {
                    AddComponent(ref rewardBand, 1f * (previousBandError - bandError));
                }
                if ((task == "stop" || task == "composite") && bandError < 0.5f)
                {
                    AddComponent(ref penaltyStopSpeed, -0.003f * horizontalVelocity.magnitude);
                }
                if (task == "composite" && !visible)
                {
                    AddComponent(ref penaltyInvisible, -0.0002f);
                }
            }
            if (task == "look")
            {
                if (!alignedOnce && visible && ViewAngle() <= 7f)
                {
                    alignedOnce = true;
                    AddComponent(ref rewardMilestone, 0.2f);
                }
            }
            if (task == "stop" && !bandEntered && bandError == 0f)
            {
                bandEntered = true;
                AddComponent(ref rewardMilestone, 0.2f);
            }
            previousCosine = cosine;
            previousDistance = distance;
            previousBandError = bandError;

            Vector3 localPosition = body.position - ArenaOrigin;
            float arenaLimit = task == "treasure" ? 11.65f
                : task == "look" || task == "approach" || task == "jump" ? 8.65f : 5.65f;
            if (Mathf.Abs(localPosition.x) > arenaLimit || Mathf.Abs(localPosition.z) > arenaLimit)
            {
                Finish(false, "out_of_bounds");
                return;
            }
            if (task == "composite" && !firstStopEntry && distance >= 1.25f && distance <= 1.75f)
            {
                firstStopEntry = true;
                firstEntryVisibleFraction = (float)visibleTicks / physicsTicks;
                if (firstEntryVisibleFraction < 0.9f)
                {
                    Finish(false, "first_entry_visibility");
                    return;
                }
            }
            if (SuccessCondition(visible, distance))
            {
                holdTicks++;
            }
            else
            {
                holdTicks = 0;
            }
            bool heldSuccess = task == "look" || task == "stop" || task == "composite";
            // 25 consecutive physics ticks at 0.02 seconds = 0.5 seconds.
            int requiredHoldTicks = 25;
            if ((!heldSuccess && holdTicks >= 1) || (heldSuccess && holdTicks >= requiredHoldTicks))
            {
                Finish(true, "success");
                return;
            }
            if (physicsTicks % 100 == 0)
            {
                LogState("checkpoint");
            }
            if (physicsTicks >= MaxDecisions() * 5)
            {
                episodeEnding = true;
                LogState("time_limit");
                EpisodeInterrupted();
            }
        }

        public override void Heuristic(in ActionBuffers actionsOut)
        {
            // Explicit environment checks only. Training never passes this CLI flag.
            var a = actionsOut.ContinuousActions;
            for (int i = 0; i < a.Length; i++) a[i] = 0f;
            if (string.IsNullOrEmpty(scriptedProbe) || layout == null || episodeEnding) return;
            Vector3 goal = Target.position;
            if (task == "jump" || task == "treasure")
            {
                Vector3 forward = Platform.transform.forward;
                Vector3 side = Platform.transform.right;
                if (task == "treasure" && probePhase < 2)
                {
                    // Detour outside the obstacles, then jump diagonally onto the top.
                    goal = probePhase == 0
                        ? ArenaOrigin + side * 2.5f + forward * 0.4f
                        : ArenaOrigin + side * 2.5f + forward *
                            (Vector3.Dot(Platform.transform.position - ArenaOrigin, forward) - 1f);
                    Vector3 d = goal - body.position; d.y = 0f;
                    if (d.magnitude < 0.3f) probePhase++;
                }
                else
                {
                    goal = task == "treasure" ? Target.position : Platform.transform.position;
                    if (jumpCount == 0 && scriptedProbe != "walk")
                    {
                        Vector3 difference = body.position - Platform.transform.position;
                        if (task == "treasure" || (Mathf.Abs(Vector3.Dot(difference, forward)) < 2.6f
                            && Mathf.Abs(Vector3.Dot(difference, side)) < 0.2f))
                            a[4] = 1f;
                    }
                }
            }
            Vector3 direction = goal - body.position; direction.y = 0f;
            Vector3 move = Vector3.ClampMagnitude(direction / 0.6f, 1f);
            if (task != "look")
            {
                a[0] = Vector3.Dot(move, YawRig.right);
                a[1] = Vector3.Dot(move, YawRig.forward);
            }
            Vector3 toward = Target.position - PolicyCamera.transform.position;
            float targetYaw = Mathf.Atan2(toward.x, toward.z) * Mathf.Rad2Deg;
            float targetPitch = -Mathf.Atan2(toward.y, new Vector2(toward.x, toward.z).magnitude) * Mathf.Rad2Deg;
            a[2] = Mathf.Clamp(Mathf.DeltaAngle(yawDegrees, targetYaw) / 9f, -1f, 1f);
            a[3] = Mathf.Clamp((targetPitch - pitchDegrees) / 9f, -1f, 1f);
        }

        private bool SuccessCondition(bool visible, float distance)
        {
            float speed = horizontalVelocity.magnitude;
            if (task == "look")
            {
                return visible && ViewAngle() <= 7f;
            }
            if (task == "approach" || task == "avoid")
            {
                // These tasks finish on the target trigger, not at a stand-off distance.
                return false;
            }
            if (task == "treasure")
            {
                // Treasure collection is handled by the target's trigger.
                return false;
            }
            if (task == "jump") return false; // Top landing is checked from contact normals.
            if (task == "stop")
            {
                return distance >= 1.25f && distance <= 1.75f && speed <= 0.2f;
            }
            if (task == "composite")
            {
                return firstStopEntry && firstEntryVisibleFraction >= 0.9f
                    && distance >= 1.25f && distance <= 1.75f
                    && speed <= 0.2f && visible && ViewAngle() <= 10f;
            }
            throw new InvalidOperationException("Unknown task: " + task);
        }

        private int MaxDecisions()
        {
            switch (task)
            {
                case "look": return 150;
                case "approach": return 200;
                case "jump": return 200;
                case "stop": return 100;
                case "avoid": return 300;
                case "treasure": return 300;
                case "composite": return 400;
                default: throw new InvalidOperationException("Unknown task: " + task);
            }
        }

        private void OnCollisionEnter(Collision collision)
        {
            CheckPlatformLanding(collision);
            if (actionStarted && !episodeEnding
                && collision.gameObject.name.StartsWith("Obstacle", StringComparison.Ordinal))
            {
                Finish(false, "collision_" + collision.gameObject.name);
            }
        }

        private void OnCollisionStay(Collision collision)
        {
            CheckPlatformLanding(collision);
        }

        private void CheckPlatformLanding(Collision collision)
        {
            if (!actionStarted || episodeEnding || Platform == null
                || collision.collider != Platform || !airborneAfterJump || jumpCount == 0)
                return;
            Vector3 local = Platform.transform.InverseTransformPoint(body.position);
            float top = Platform.bounds.max.y;
            if (Mathf.Abs(local.x) >= 0.5f - 0.1f / 3f
                || Mathf.Abs(local.z) >= 0.5f - 0.1f / 2.4f
                || Mathf.Abs(body.position.y - 1f - top) > 0.08f || body.linearVelocity.y > 0.3f)
                return;
            foreach (ContactPoint contact in collision.contacts)
            {
                if (contact.normal.y > 0.7f && Mathf.Abs(contact.point.y - top) < 0.08f)
                {
                    landedOnPlatform = true;
                    if (task == "jump") Finish(true, "success");
                    return;
                }
            }
        }

        private void OnTriggerEnter(Collider other)
        {
            CollectTarget(other);
        }

        private void OnTriggerStay(Collider other)
        {
            // A first policy action may arrive while the player is touching the panel.
            CollectTarget(other);
        }

        private void CollectTarget(Collider other)
        {
            if ((task == "approach" || task == "avoid" || task == "treasure")
                && actionStarted && !episodeEnding && other.transform == Target)
            {
                if (task == "treasure" && layout.has_platform &&
                    (!landedOnPlatform || body.position.y - 1f < Platform.bounds.max.y - 0.08f))
                    return;
                Finish(true, "success");
            }
        }

        private void Finish(bool success, string reason)
        {
            if (episodeEnding)
            {
                return;
            }
            episodeEnding = true;
            AddComponent(ref rewardTerminal, success ? 1f : -1f);
            LogState(reason);
            EndEpisode();
        }

        private void AddComponent(ref float accumulator, float amount)
        {
            AddReward(amount);
            accumulator += amount;
        }

        private void LogState(string reason)
        {
            Debug.Log("TPS_EPISODE reward_version=" + (task == "treasure" ? "treasure-gaze-v0.4" : RewardVersion)
                + " task=" + task + " layout=" + layout.layout_id
                + " head=" + HeadId + " color_variant=" + ColorVariant
                + " jumps=" + jumpCount + " landed_platform=" + landedOnPlatform
                + " y=" + body.position.y
                + " reason=" + reason + " ticks=" + physicsTicks
                + " hold_ticks=" + holdTicks
                + " displacement_m=" + Vector2.Distance(
                    new Vector2(body.position.x, body.position.z),
                    new Vector2(startPosition.x, startPosition.z))
                + " x=" + body.position.x + " z=" + body.position.z
                + " distance=" + TargetDistance() + " speed=" + horizontalVelocity.magnitude
                + " angle=" + ViewAngle() + " visibility=" + visibleTicks + "/" + physicsTicks
                + " treasure_found=" + treasureFound
                + " treasure_first_seen_tick=" + treasureFirstSeenTick
                + " first_entry=" + firstStopEntry
                + " first_entry_visibility=" + firstEntryVisibleFraction
                + " reward_time=" + rewardTime
                + " reward_view=" + rewardView
                + " reward_distance=" + rewardDistance
                + " reward_band=" + rewardBand
                + " penalty_look_movement=" + penaltyLookMovement
                + " penalty_stop_speed=" + penaltyStopSpeed
                + " penalty_invisible=" + penaltyInvisible
                + " reward_terminal=" + rewardTerminal
                + " reward_milestone=" + rewardMilestone
                + " penalty_band_distance=" + penaltyBandDistance
                + " penalty_look_command=" + penaltyLookCommand
                + " penalty_stop_view=" + penaltyStopView
                + " penalty_stop_camera_command=" + penaltyStopCameraCommand
                + " penalty_camera_command=" + penaltyCameraCommand
                + " obstacle_motion=" + (movingObstacles ? "moving" : "static")
                + " reward_view_reference_cosine=" + ViewCosineForReward());
        }

        private float TargetDistance()
        {
            Vector3 delta = Target.position - body.position;
            delta.y = 0f;
            return delta.magnitude;
        }

        private static float BandError(float distance)
        {
            return Mathf.Max(1.25f - distance, 0f, distance - 1.75f);
        }

        private float ViewAngle()
        {
            Vector3 direction = Target.position - PolicyCamera.transform.position;
            return Vector3.Angle(PolicyCamera.transform.forward, direction);
        }

        private float ViewCosine()
        {
            return Mathf.Cos(ViewAngle() * Mathf.Deg2Rad);
        }

        private float ViewCosineForReward()
        {
            // Movement is allowed in look; shaping follows the actual camera.
            return ViewCosine();
        }

        private bool TargetVisible()
        {
            Vector3 point = Target.position;
            Vector3 viewport = PolicyCamera.WorldToViewportPoint(point);
            if (viewport.z <= 0f || viewport.x < 0.1f || viewport.x > 0.9f
                || viewport.y < 0.1f || viewport.y > 0.9f)
            {
                return false;
            }
            Vector3 direction = point - PolicyCamera.transform.position;
            RaycastHit hit;
            return Physics.Raycast(PolicyCamera.transform.position, direction.normalized,
                out hit, direction.magnitude + 0.01f, Physics.DefaultRaycastLayers,
                QueryTriggerInteraction.Collide)
                && hit.transform == Target;
        }

        private void MoveObstacles()
        {
            Vector3 route = Target.position - startPosition;
            route.y = 0f;
            Vector3 sideways = new Vector3(-route.z, 0f, route.x).normalized;
            for (int index = 0; index < Obstacles.Length; index++)
            {
                if (!Obstacles[index].activeSelf)
                {
                    continue;
                }
                float sign = index % 2 == 0 ? 1f : -1f;
                float offset = 0.65f * Mathf.Sin(2f * Mathf.PI * 0.30f
                    * physicsTicks * Time.fixedDeltaTime) * sign;
                Vector3 destination = obstacleBases[index] + sideways * offset;
                Vector3 local = destination - ArenaOrigin;
                destination.x = ArenaOrigin.x + Mathf.Clamp(local.x, -8.2f, 8.2f);
                destination.z = ArenaOrigin.z + Mathf.Clamp(local.z, -8.2f, 8.2f);
                Obstacles[index].transform.position = destination;
            }
            Physics.SyncTransforms();
        }

        private void UpdateCameraRig()
        {
            YawRig.localRotation = Quaternion.Euler(0f, yawDegrees, 0f);
            PitchRig.localRotation = Quaternion.Euler(pitchDegrees, 0f, 0f);
        }

        private static string ArgumentValue(string name, string fallback)
        {
            foreach (string argument in Environment.GetCommandLineArgs())
            {
                if (argument.StartsWith(name + "=", StringComparison.Ordinal))
                {
                    return argument.Substring(name.Length + 1);
                }
            }
            return fallback;
        }
    }
}
