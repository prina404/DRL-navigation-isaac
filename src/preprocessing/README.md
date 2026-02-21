# InteriorAgent preprocessing

-- TODO: script to automate dataset download

To process all 25 InteriorAgent's environments, set the correct dataset folder into src/cfg/CFG.py, then run
```bash
$ cd src/preprocessing
$ python3 preprocess_interioragent.py
```

## Steps performed
- Create two root folders: `static_objects/` and `dynamic_objects/`. This is needed in order to correctly track dynamic objects with the MultiMeshRayCaster Lidar.
- Disable all unnecessary Rigid Bodies: everything becomes a static collider, apart from doors (or other objects in the future).
- Simplify collisions with convex hulls and convex decomposition.
- Preprocess doors according to the categories defined in the `src/cfg/interioragetn_preprocessing.yaml` configuration file.
- Compute the occupancy grid map image and .yaml which will be used during learning.
## Door USD

There are three kinds of doors in the dataset (manually labeled):
- Standard hinged doors
- Sliding doors
- Perimeter doors

In the dataset, all doors apart from perimeter ones are disabled (invisible and without collisions), so during preprocessing we enable them back and add a DriveAPI motor to control them during the simulation.

Note that sliding doors are currently kept disabled, and only the hinged doors can be moved in a (0°, 90°) range.

### Door hierarchy is always:
```
other/
└── door_00xx/
    └── Meshes/
        └── door_00xx/
            ├── constraint_1/        (RevoluteJoint at hinge position + DriveAPI)
            ├── constraint_2/        (optional second hinge marker)
            ├── group_0000/          (Door Frame — RigidBody: Static + Collider)
            │   └── <frame meshes>
            ├── group_0001/         (Door Leaf — RigidBody: Dynamic + Collider)
            │   └── <door meshes>
            └── group_00xx/
                └── <remainig details>
```

Regex for prims preprocessing
 ---

 - Top level **Meshes/door_00xx**: `/Meshes/door_\d+[/]*$`
 - Door frame **group_0000**: `/door_\d+/group_0000[/]*$`
 - Door body **group_0001**: `/door_\d+/group_0001[/]*$`
 - Lower hinge : `/door_\d+/physics_constraint_0001[/]*$`
