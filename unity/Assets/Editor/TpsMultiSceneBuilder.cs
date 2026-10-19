using System;
using System.IO;
using TpsTransfer;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Policies;
using Unity.MLAgents.Sensors;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace TpsTransfer.Editor
{
    public static class TpsMultiSceneBuilder
    {
        [Serializable]
        private sealed class ColorVariant
        {
            public string id;
            public float[] target_rgb;
            public float[] avatar_rgb;
        }

        [Serializable]
        private sealed class ColorProtocol
        {
            public ColorVariant[] variants;
        }

        private static ColorVariant[] colorVariants;
        private static PhysicsMaterial jumpContact;
        private const string PretrainScene = "Assets/Scenes/TpsPretrain.unity";
        private const string CompositeScene = "Assets/Scenes/TpsComposite.unity";
        private const string PretrainThreeScene = "Assets/Scenes/TpsPretrainThree.unity";
        private const string PretrainTwoScene = "Assets/Scenes/TpsPretrainTwo.unity";
        private const string TreasureScene = "Assets/Scenes/TpsTreasure.unity";
        private static readonly string[] PrimitiveTasks = { "look", "approach", "stop", "avoid" };
        private static readonly string[] TreasureSourceTasks = { "look", "approach", "avoid" };
        private static readonly string[] GazeSourceTasks = { "look", "approach" };

        public static void BuildJumpSingleColorLinuxPlayers()
        {
            jumpContact = new PhysicsMaterial("JumpContact") { dynamicFriction = 0f, staticFriction = 0f,
                frictionCombine = PhysicsMaterialCombine.Minimum, bounciness = 0f };
            const string contactPath = "Assets/Materials/V10JumpContact.physicMaterial";
            AssetDatabase.DeleteAsset(contactPath);
            AssetDatabase.CreateAsset(jumpContact, contactPath);
            CreateScene("Assets/Scenes/TpsPretrainJump.unity", new[] { "look", "approach", "jump" }, 18);
            Build("Assets/Scenes/TpsPretrainJump.unity", "Build/TpsPretrainJump.x86_64");
            BuildTreasureLinuxPlayer();
        }

        public static void BuildV10TreasureLinuxPlayer()
        {
            jumpContact = AssetDatabase.LoadAssetAtPath<PhysicsMaterial>("Assets/Materials/V10JumpContact.physicMaterial");
            if (jumpContact == null) throw new InvalidDataException("Missing v0.10 contact material");
            BuildTreasureLinuxPlayer();
        }

        [MenuItem("TPS/Build Linux color variant players")]
        public static void BuildColorVariantLinuxPlayers()
        {
            var protocol = JsonUtility.FromJson<ColorProtocol>(File.ReadAllText(
                "Assets/StreamingAssets/primitive_color_variants.json"));
            colorVariants = protocol.variants;
            if (colorVariants == null || colorVariants.Length != 3)
                throw new InvalidDataException("Expected three primitive color variants");
            var heads = new string[6];
            for (int taskIndex = 0; taskIndex < 2; taskIndex++)
                for (int variant = 0; variant < 3; variant++)
                    heads[taskIndex * 3 + variant] = GazeSourceTasks[taskIndex]
                        + "_" + colorVariants[variant].id;
            const string scene = "Assets/Scenes/TpsPretrainColors.unity";
            CreateScene(scene, heads, 36);
            Build(scene, "Build/TpsPretrainColors.x86_64");
            BuildTreasureLinuxPlayer();
        }

        [MenuItem("TPS/Build Linux edge appearance players")]
        public static void BuildEdgeAppearanceLinuxPlayers()
        {
            BuildPretrainTwoLinuxPlayer();
            BuildTreasureLinuxPlayer();
            BuildGazePreviewLinuxPlayers();
        }

        [MenuItem("TPS/Build Linux pretraining player")]
        public static void BuildPretrainingLinuxPlayer()
        {
            if (!File.Exists(PretrainScene))
            {
                CreateScene(PretrainScene, PrimitiveTasks, 16);
            }
            Build(PretrainScene, "Build/TpsPretrain.x86_64");
        }

        [MenuItem("TPS/Build Linux composite player")]
        public static void BuildCompositeLinuxPlayer()
        {
            CreateScene(CompositeScene, new[] { "composite" }, 16);
            Build(CompositeScene, "Build/TpsComposite.x86_64");
        }

        [MenuItem("TPS/Recreate multi scenes")]
        public static void RecreateMultiScenes()
        {
            CreateScene(PretrainScene, PrimitiveTasks, 16);
            CreateScene(CompositeScene, new[] { "composite" }, 16);
        }

        [MenuItem("TPS/Build Linux three-task pretraining player")]
        public static void BuildPretrainThreeLinuxPlayer()
        {
            CreateScene(PretrainThreeScene, TreasureSourceTasks, 12);
            Build(PretrainThreeScene, "Build/TpsPretrainThree.x86_64");
        }

        [MenuItem("TPS/Build Linux two-task gaze pretraining player")]
        public static void BuildPretrainTwoLinuxPlayer()
        {
            CreateScene(PretrainTwoScene, GazeSourceTasks, 12);
            Build(PretrainTwoScene, "Build/TpsPretrainTwo.x86_64");
        }

        [MenuItem("TPS/Build Linux gaze preview players")]
        public static void BuildGazePreviewLinuxPlayers()
        {
            CreateScene("Assets/Scenes/TpsLookPreview.unity", new[] { "look" }, 1);
            Build("Assets/Scenes/TpsLookPreview.unity", "Build/TpsLookPreview.x86_64");
            CreateScene("Assets/Scenes/TpsApproachPreview.unity", new[] { "approach" }, 1);
            Build("Assets/Scenes/TpsApproachPreview.unity", "Build/TpsApproachPreview.x86_64");
            CreateScene("Assets/Scenes/TpsTreasurePreview.unity", new[] { "treasure" }, 1);
            Build("Assets/Scenes/TpsTreasurePreview.unity", "Build/TpsTreasurePreview.x86_64");
        }

        [MenuItem("TPS/Build Linux treasure player")]
        public static void BuildTreasureLinuxPlayer()
        {
            CreateScene(TreasureScene, new[] { "treasure" }, 16);
            Build(TreasureScene, "Build/TpsTreasure.x86_64");
        }

        private static void CreateScene(string path, string[] tasks, int arenaCount)
        {
            if (!File.Exists("Assets/StreamingAssets/pilot_layouts.json"))
            {
                throw new FileNotFoundException("pilot layout manifest is missing");
            }
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            bool pretraining = tasks.Length > 1;
            for (int index = 0; index < arenaCount; index++)
            {
                string task = tasks[index % tasks.Length];
                var origin = new Vector3((index % 4) * 100f, 0f, (index / 4) * 100f);
                CreateArena(index, origin, task, pretraining, tasks.Length, arenaCount);
            }
            var lightObject = new GameObject("DirectionalLight");
            var light = lightObject.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.2f;
            lightObject.transform.rotation = Quaternion.Euler(45f, -30f, 0f);
            EditorSceneManager.SaveScene(scene, path);
            AssetDatabase.SaveAssets();
        }

        private static void CreateArena(int index, Vector3 origin, string head,
            bool pretraining, int taskCount, int arenaCount)
        {
            string task = head;
            ColorVariant palette = null;
            if (head.Contains("_color"))
            {
                task = head.Substring(0, head.LastIndexOf('_'));
                string variantId = head.Substring(head.LastIndexOf('_') + 1);
                palette = Array.Find(colorVariants, variant => variant.id == variantId);
                if (palette == null) throw new InvalidDataException("Unknown color head: " + head);
            }
            var root = new GameObject("Arena" + index);
            root.transform.position = origin;
            var floor = GameObject.CreatePrimitive(PrimitiveType.Cube);
            floor.name = "Floor";
            floor.transform.SetParent(root.transform, false);
            floor.transform.localPosition = new Vector3(0f, -0.1f, 0f);
            float floorSize = task == "treasure" || task == "look" || task == "approach" || task == "jump"
                ? 18f : 12f;
            floor.transform.localScale = new Vector3(floorSize, 0.2f, floorSize);
            SetColor(floor, new Color(0.48f, 0.5f, 0.5f));

            var target = GameObject.CreatePrimitive(PrimitiveType.Cube);
            target.name = task == "treasure" ? "Treasure" : "VisibleTarget";
            target.transform.SetParent(root.transform, false);
            target.transform.localPosition = new Vector3(0f, 1.6f, 5f);
            target.transform.localScale = new Vector3(1f, 2f, 0.4f);
            bool downstreamAppearance = task == "treasure" || task == "composite";
            target.GetComponent<MeshFilter>().sharedMesh = downstreamAppearance
                ? DiamondMesh() : PrimitiveMesh(PrimitiveType.Sphere);
            SetColor(target, downstreamAppearance
                ? new Color(1f, 0.38f, 0.03f)
                : palette == null ? new Color(1f, 0.38f, 0.03f) : Rgb(palette.target_rgb),
                palette == null ? null : "VisibleTarget_" + palette.id);
            if (task == "treasure" || task == "approach" || task == "avoid")
            {
                // Use the same contact volume while changing visual appearance.
                target.GetComponent<BoxCollider>().isTrigger = true;
            }

            BoxCollider platformCollider = null;
            if (task == "jump" || task == "treasure")
            {
                var platform = GameObject.CreatePrimitive(PrimitiveType.Cube);
                platform.name = "JumpPlatform";
                platform.transform.SetParent(root.transform, false);
                platform.transform.localPosition = new Vector3(0f, 0.5f, 3f);
                platform.transform.localScale = new Vector3(3f, 1f, 2.4f);
                SetColor(platform, new Color(1f, 0.38f, 0.03f));
                platformCollider = platform.GetComponent<BoxCollider>();
                if (task == "jump") target.GetComponent<Renderer>().enabled = false;
                if (task == "jump") target.GetComponent<Collider>().enabled = false;
            }

            var obstacles = new GameObject[3];
            for (int obstacleIndex = 0; obstacleIndex < obstacles.Length; obstacleIndex++)
            {
                var obstacle = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                obstacle.name = "Obstacle" + obstacleIndex;
                obstacle.transform.SetParent(root.transform, false);
                obstacle.transform.localPosition = new Vector3(0f, 0.5f, 20f);
                SetColor(obstacle, new Color(0.25f, 0.25f, 0.28f));
                obstacle.SetActive(false);
                obstacles[obstacleIndex] = obstacle;
            }

            var player = GameObject.CreatePrimitive(PrimitiveType.Capsule);
            player.name = "TaskAgent" + index;
            player.transform.SetParent(root.transform, false);
            player.transform.localPosition = new Vector3(0f, 1f, 0f);
            player.transform.localScale = new Vector3(0.7f, 1f, 0.7f);
            player.GetComponent<Renderer>().enabled = false;
            var avatar = GameObject.CreatePrimitive(downstreamAppearance
                ? PrimitiveType.Cube : PrimitiveType.Cylinder);
            avatar.name = downstreamAppearance ? "TreasureAvatar" : "AtomicAvatar";
            avatar.transform.SetParent(player.transform, false);
            avatar.transform.localScale = downstreamAppearance
                ? new Vector3(1f, 2f, 1f) : Vector3.one;
            UnityEngine.Object.DestroyImmediate(avatar.GetComponent<Collider>());
            SetColor(avatar, downstreamAppearance
                ? new Color(0.03f, 0.82f, 0.9f)
                : palette == null ? new Color(0.03f, 0.82f, 0.9f) : Rgb(palette.avatar_rgb),
                palette == null ? null : "AtomicAvatar_" + palette.id);
            var body = player.AddComponent<Rigidbody>();
            body.mass = 1f;
            body.constraints = RigidbodyConstraints.FreezeRotation;
            body.interpolation = RigidbodyInterpolation.None;
            body.collisionDetectionMode = CollisionDetectionMode.ContinuousDynamic;
            player.GetComponent<Collider>().sharedMaterial = jumpContact;

            var yawRig = new GameObject("YawRig").transform;
            yawRig.SetParent(player.transform, false);
            var pitchRig = new GameObject("PitchRig").transform;
            pitchRig.SetParent(yawRig, false);
            var cameraObject = new GameObject("PolicyCamera");
            cameraObject.transform.SetParent(pitchRig, false);
            if (task == "look" || task == "approach" || task == "avoid"
                || task == "treasure" || task == "composite" || task == "jump")
            {
                // Share one over-shoulder view across source and downstream tasks.
                // The offset keeps the target visible beside the avatar and the
                // camera position independent of pitch, clear of the floor.
                pitchRig.localPosition = new Vector3(1.5f, 2.0f, -3.6f);
                cameraObject.transform.localPosition = Vector3.zero;
            }
            else
            {
                cameraObject.transform.localPosition = new Vector3(0f, 2.2f, -3.5f);
            }
            var camera = cameraObject.AddComponent<Camera>();
            camera.fieldOfView = 70f;
            camera.aspect = 1f;
            camera.nearClipPlane = 0.1f;
            camera.farClipPlane = 50f;
            camera.clearFlags = CameraClearFlags.SolidColor;
            camera.backgroundColor = new Color(0.55f, 0.75f, 0.95f);
            if (index == 0)
            {
                cameraObject.AddComponent<AudioListener>();
            }

            var sensor = player.AddComponent<CameraSensorComponent>();
            sensor.Width = 224;
            sensor.Height = 224;
            sensor.Grayscale = false;
            sensor.CompressionType = SensorCompressionType.PNG;
            sensor.ObservationStacks = 1;
            sensor.SensorName = "policy_rgb";
            sensor.Camera = camera;

            var behavior = player.AddComponent<BehaviorParameters>();
            behavior.BehaviorName = "TPS_" + head;
            behavior.BehaviorType = BehaviorType.Default;
            behavior.BrainParameters.ActionSpec = ActionSpec.MakeContinuous(5);
            behavior.BrainParameters.VectorObservationSize = 0;

            var agent = player.AddComponent<TpsTaskAgent>();
            var decisions = player.AddComponent<Unity.MLAgents.DecisionRequester>();
            decisions.DecisionPeriod = 5;
            decisions.DecisionStep = 0;
            decisions.TakeActionsBetweenDecisions = false;
            agent.YawRig = yawRig;
            agent.PitchRig = pitchRig;
            agent.PolicyCamera = camera;
            agent.Target = target.transform;
            agent.Platform = platformCollider;
            agent.Obstacles = obstacles;
            agent.TaskOverride = task;
            agent.HeadId = head;
            agent.ColorVariant = palette == null ? "none" : palette.id;
            agent.ArenaOrigin = origin;
            agent.LayoutStartOffset = pretraining ? index / taskCount : index;
            agent.LayoutStride = pretraining ? arenaCount / taskCount : arenaCount;
        }

        private static void Build(string scene, string output)
        {
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { scene },
                locationPathName = output,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.CleanBuildCache
            });
            if (report.summary.result != BuildResult.Succeeded)
            {
                throw new Exception("TPS multi-arena player build failed: " + report.summary.result);
            }
        }

        private static Mesh PrimitiveMesh(PrimitiveType type)
        {
            var temporary = GameObject.CreatePrimitive(type);
            var mesh = temporary.GetComponent<MeshFilter>().sharedMesh;
            UnityEngine.Object.DestroyImmediate(temporary);
            return mesh;
        }

        private static Mesh DiamondMesh()
        {
            if (!AssetDatabase.IsValidFolder("Assets/Meshes"))
                AssetDatabase.CreateFolder("Assets", "Meshes");
            const string path = "Assets/Meshes/TreasureDiamond.asset";
            var mesh = AssetDatabase.LoadAssetAtPath<Mesh>(path);
            if (mesh != null) return mesh;
            mesh = new Mesh { name = "TreasureDiamond" };
            mesh.vertices = new[] {
                new Vector3(0f, 0.5f, 0f), new Vector3(0f, -0.5f, 0f),
                new Vector3(0f, 0f, 0.5f), new Vector3(0.5f, 0f, 0f),
                new Vector3(0f, 0f, -0.5f), new Vector3(-0.5f, 0f, 0f)
            };
            mesh.triangles = new[] {
                0, 2, 3, 0, 3, 4, 0, 4, 5, 0, 5, 2,
                1, 3, 2, 1, 4, 3, 1, 5, 4, 1, 2, 5
            };
            mesh.RecalculateNormals();
            mesh.RecalculateBounds();
            AssetDatabase.CreateAsset(mesh, path);
            return mesh;
        }

        private static Color Rgb(float[] rgb)
        {
            if (rgb == null || rgb.Length != 3)
                throw new InvalidDataException("Expected RGB triplet");
            return new Color(rgb[0], rgb[1], rgb[2]);
        }

        private static void SetColor(GameObject gameObject, Color color, string materialKey = null)
        {
            if (!AssetDatabase.IsValidFolder("Assets/Materials"))
            {
                AssetDatabase.CreateFolder("Assets", "Materials");
            }
            string materialName = materialKey ?? (gameObject.name.StartsWith("TaskAgent", StringComparison.Ordinal)
                ? "TaskAgent" : gameObject.name);
            string path = "Assets/Materials/" + materialName + ".mat";
            var material = AssetDatabase.LoadAssetAtPath<Material>(path);
            if (material == null)
            {
                material = new Material(Shader.Find("Standard"));
                AssetDatabase.CreateAsset(material, path);
            }
            material.color = color;
            if (gameObject.name == "VisibleTarget" || gameObject.name == "Treasure"
                || gameObject.name == "AtomicAvatar" || gameObject.name == "TreasureAvatar")
            {
                material.EnableKeyword("_EMISSION");
                material.SetColor("_EmissionColor", color * 0.25f);
            }
            gameObject.GetComponent<Renderer>().sharedMaterial = material;
        }
    }
}
