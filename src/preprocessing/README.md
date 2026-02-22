# InteriorAgent preprocessing

-- TODO: script to automate dataset download

To process all 25 InteriorAgent's environments, set the correct dataset folder into `src/cfg/CFG.py`, then run
```bash
$ cd src/preprocessing
$ python3 preprocess_interioragent.py
```

## Steps performed
- In the USD file, create two root folders: `static_objects/` and `dynamic_objects/`. This is needed in order to correctly track dynamic objects with the MultiMeshRayCaster Lidar.
- Disable all unnecessary Rigid Bodies: everything becomes a static collider, apart from doors (or other objects in the future).
- Simplify collisions with convex hulls and convex decomposition.
- Preprocess doors according to the categories defined in the `src/cfg/interioragetn_preprocessing.yaml` configuration file.
- Compute the occupancy grid map image and .yaml which will be used during learning.

## Door properties in the original dataset

I have separated the doors in three categories:
- Standard hinged door
- Sliding door
- Perimeter door

In the dataset, all doors apart from perimeter ones are disabled (invisible and without collisions), so during preprocessing we enable them back and add a DriveAPI motor to control them during the simulation.

Note that sliding doors are currently kept disabled, and only the hinged doors can be moved in a (0°, 90°) range.

### Door hierarchy is always:
```
...
other/
└── door_00xx/                      (has RigidBodyAPI)
    └── Meshes/
        └── door_00xx/
            ├── constraint_1/        
            ├── constraint_2/        
            ├── group_0000/         (Has RigidBodyAPI with kinematic=True)
            │   └── <frame meshes>
            └── group_000x/
                └── <body meshes>   (prims disabled + no physics API applied)
```

After preprocessing, the doors become:

```
...
other/
└── door_00xx/                       (no physics API applied)
    └── Meshes/
        └── door_00xx/
            ├── constraint_1/        
            │   └── RevoluteJoint    (RevoluteJointAPI + DriveAPI)
            ├── constraint_2/        
            │   └── RevoluteJoint    (RevoluteJointAPI + DriveAPI)
            ├── group_0000/          (Has RigidBodyAPI with kinematic=True)
            │   └── <frame meshes>
            └── group_000x/          (Has RigidBodyAPI with kinematic=False)
                └── <body meshes>
```

Finally, the doors can be controlled as follows:
```python
root_prim = stage.GetPrimAtPath(root_path)

for prim in Usd.PrimRange(root_prim):
    if prim.IsValid() and prim.GetTypeName() == "PhysicsRevoluteJoint":
            door_drive = UsdPhysics.DriveAPI.Apply(door_prim, UsdPhysics.Tokens.angular)
            door_drive.CreateTargetPositionAttr().Set(45) # set door at 45°
```


Useful regex for door preprocessing
 ---
 - Root xform **other/door_00xx**: `/other/door_\d+[/]*$`
 - Top level mesh xform **Meshes/door_00xx**: `/Meshes/door_\d+[/]*$`
 - Door frame **group_0000**: `/door_\d+/group_0000[/]*$`
 - Door body **group_000x**: `/door_\d+/group_000[1-9][/]*$`
 - Hinges : `/door_\d+/physics_constraint_000[1-9[/]*$`
