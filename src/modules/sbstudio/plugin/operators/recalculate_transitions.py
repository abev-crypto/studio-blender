from dataclasses import dataclass
from enum import Enum
from functools import partial
from math import inf
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, Union, cast

import bpy
from bpy.types import Collection, Mesh, MeshVertex, Object

from bpy.props import EnumProperty

from sbstudio.api.errors import SkybrushStudioAPIError
from sbstudio.api.types import Mapping
from sbstudio.errors import SkybrushStudioError
from sbstudio.plugin.actions import (
    cleanup_actions_for_object,
    ensure_action_exists_for_object,
)
from sbstudio.plugin.api import call_api_from_blender_operator, get_api
from sbstudio.plugin.constants import Collections
from sbstudio.plugin.keyframes import set_keyframes
from sbstudio.plugin.model.formation import (
    get_markers_and_related_objects_from_formation,
    get_world_coordinates_of_markers_from_formation,
)
from sbstudio.plugin.model.storyboard import (
    Storyboard,
    StoryboardEntry,
    StoryboardEntryPurpose,
)
from sbstudio.plugin.tasks.safety_check import invalidate_caches
from sbstudio.plugin.utils import create_internal_id
from sbstudio.plugin.utils.evaluator import create_position_evaluator
from sbstudio.plugin.utils.transition import (
    create_transition_constraint_between,
    find_transition_constraint_between,
    set_constraint_name_from_storyboard_entry,
)
from sbstudio.utils import constant

from .base import StoryboardOperator

__all__ = ("RecalculateTransitionsOperator",)


class InfluenceCurveTransitionType(Enum):
    """Possible types of a transition phase of an influence curve."""

    LINEAR = "linear"
    SMOOTH_FROM_LEFT = "smoothFromLeft"
    SMOOTH_FROM_RIGHT = "smoothFromRight"
    SMOOTH = "smooth"


@dataclass
class InfluenceCurveDescriptor:
    """Dataclass that describes how the influence curve of a constraint should
    look like.

    We currently work with one type of influence curve at the moment. The curve
    starts from zero at the start of the scene, stays zero until (and including)
    a _windup start frame_, then transitions to 1 until (and including) a
    _start frame_. The transition can be linear, smooth from the left,
    smooth from the right or completely smooth. The curve will then either stay
    1 indefinitely, or stay 1 until a designated _end frame_.

    The key points of the influence curve are described by this data class. It
    also provides a method to apply these points to an existing constraint.
    """

    scene_start_frame: int
    """The start frame of the entire scene."""

    windup_start_frame: Optional[int]
    """The windup start frame, i.e. the _last_ frame when the influence curve
    should still be zero before winding up to full influence. `None` means that
    it is the same as the start frame of the scene.

    When this frame is earlier than the start frame of the scene, it is assumed
    to be equal to the start frame of the scene.
    """

    start_frame: int
    """The first frame when the influence should become equal to 1. Must be
    larger than the windup start frame; when it is smaller or equal, it is
    assumed to be one larger than the windup start frame.
    """

    end_frame: Optional[int] = None
    """The last frame when the influence is still equal to 1; ``None`` means
    that the influence curve stays 1 infinitely.

    The end frame must be larger than or equal to the start frame; when it is
    smaller, it is assumed to be equal to the start frame.
    """

    windup_type: InfluenceCurveTransitionType = InfluenceCurveTransitionType.SMOOTH
    """The type of the windup transition."""

    def __init__(
        self,
        scene_start_frame: int,
        windup_start_frame: Optional[int],
        start_frame: int,
        end_frame: Optional[int] = None,
        windup_type: InfluenceCurveTransitionType = InfluenceCurveTransitionType.SMOOTH,
    ):
        # Note that explicit __init__() method implementation is needed to
        # ensure that int type arguments are truly ints
        self.scene_start_frame = round(scene_start_frame)
        self.windup_start_frame = (
            None if windup_start_frame is None else round(windup_start_frame)
        )
        self.start_frame = round(start_frame)
        self.end_frame = None if end_frame is None else round(end_frame)
        self.windup_type = windup_type

    def apply(self, object, data_path: str) -> None:
        """Applies the influence curve descriptor to the given data path of the
        given Blender object by updating the keyframes appropriately.

        Parameters:
            object: the object on which the keyframes are to be set
            data_path: the data path to use
        """
        # Special case: if the start frame is the start of the scene, it means
        # that this is the first transition in the timeline. In this case we
        # need to start with an influence of 1 to ensure that the drone do not
        # jump around in the first frame of the show if their associated
        # takeoff marker is at a different position from the position of the
        # drone itself

        is_first = self.scene_start_frame == self.start_frame
        keyframes: List[Tuple[int, float]] = [
            (self.scene_start_frame - (1 if is_first else 0), 0.0)
        ]

        # Hold current influence value until the start of the windup
        if (
            not is_first
            and self.windup_start_frame is not None
            and self.windup_start_frame > self.scene_start_frame
        ):
            keyframes.append((self.windup_start_frame, keyframes[-1][1]))

        # Ramp up to 1 at the start frame
        frame = max(self.start_frame, keyframes[-1][0] + 1)
        start_of_transition = len(keyframes) - 1
        keyframes.append((frame, 1.0))

        # Add a keyframe at the end frame
        if self.end_frame is not None:
            end_frame = max(self.end_frame, frame)
            if end_frame > frame:
                keyframes.append((end_frame, 1.0))

            # Do not wind the constraint down to zero after the end frame; it
            # makes it harder to remove storyboard entries from the middle of
            # the storyboard as the end frame of the previous constraint would
            # have to be adjusted
            # keyframes.append((end_frame + 1, 0.0))

        keyframe_objs = set_keyframes(
            object,
            data_path,
            keyframes,
            clear_range=(None, inf),
            interpolation="LINEAR",
        )

        if self.windup_type != InfluenceCurveTransitionType.LINEAR:
            kf = keyframe_objs[start_of_transition]
            kf.interpolation = "BEZIER"
            if self.windup_type == InfluenceCurveTransitionType.SMOOTH_FROM_RIGHT:
                kf.handle_right_type = "VECTOR"
            else:
                kf.handle_right_type = "AUTO_CLAMPED"
            if self.windup_type == InfluenceCurveTransitionType.SMOOTH_FROM_LEFT:
                kf.handle_left_type = "VECTOR"
            else:
                kf.handle_left_type = "AUTO_CLAMPED"


class _LazyFormationTargetList:
    """Helper object that takes a reference to a storyboard entry, and provides
    a ``find()`` method for it to look up the index of an object within the
    formation, falling back to a default value when the object is not in the
    collection.
    """

    _formation: Optional[Collection] = None
    """The formation of the storyboard entry."""

    _items: Optional[List[Union[Object, MeshVertex]]] = None

    def __init__(self, entry: Optional[StoryboardEntry]):
        self._formation = entry.formation if entry else None

    def find(self, item, *, default: int = 0) -> int:
        if item is None:
            return default

        if self._items is None:
            self._items = self._validate_items()

        try:
            return self._items.index(item)
        except ValueError:
            return default

    def _validate_items(self) -> List[Union[Object, MeshVertex]]:
        if self._formation is None:
            return []
        else:
            return [
                v
                for v, _ in get_markers_and_related_objects_from_formation(
                    self._formation
                )
            ]


def get_coordinates_of_formation(formation, *, frame: int) -> List[Tuple[float, ...]]:
    """Returns the coordinates of all the markers in the given formation at the
    given frame as a list of triplets.
    """
    return [
        tuple(pos)
        for pos in get_world_coordinates_of_markers_from_formation(
            formation, frame=frame
        )
    ]


def calculate_mapping_for_transition_into_storyboard_entry(
    entry: StoryboardEntry, source, *, num_targets: int
) -> Mapping:
    """Calculates the mapping of source points (i.e. the current positions
    of the drones) and target points (i.e. marker positions for a storyboard
    entry).

    This function must be called only if the storyboard entry contains a
    formation (i.e. it is not a free segment).

    Parameters:
        entry: the storyboard entry
        source: the list of source points to consider
        num_targets: number of target points that the drones should be
            mapped to; used for manual mapping so we can avoid querying
            all the coordinates if they are not needed

    Returns:
        a list where the i-th element contains the index of the target point
        that the i-th drone was matched to, or ``None`` if the drone was
        left unmatched

    Raises:
        SkybrushStudioAPIError: if an error happens while querying the
            remote API that calculates the mapping
    """
    formation = entry.formation
    if formation is None:
        raise RuntimeError(
            "mapping function called for storyboard entry with no formation"
        )

    num_drones = len(source)

    result: Mapping = [None] * num_drones

    # We need to create a transition between the places where the drones
    # are at the end of the previous formation and the points of the
    # current formation
    if entry.transition_type == "AUTO":
        # Auto mapping with our API
        target = get_coordinates_of_formation(formation, frame=entry.frame_start)
        try:
            match, clearance = get_api().match_points(source, target, radius=0)
        except Exception as ex:
            if not isinstance(ex, SkybrushStudioAPIError):
                raise SkybrushStudioAPIError from ex
            else:
                raise

        # At this point we have the inverse mapping: match[i] tells the
        # index of the drone that the i-th target point was matched to, or
        # ``None`` if the target point was left unmatched. We need to invert
        # the mapping
        for target_index, drone_index in enumerate(match):
            if drone_index is not None:
                result[drone_index] = target_index

    else:
        # Manual mapping
        length = min(num_drones, num_targets)
        result[:length] = range(length)

    return result


def _vertex_index_to_vertex_group_name(index: int) -> str:
    """Converts a vertex index to the preferred name of a vertex group that holds
    this vertex only.

    Vertex groups holding a single vertex are required to make a single vertex
    become a transition target in a mesh-based formation.
    """
    return create_internal_id(f"Vertex {index}")


def _vertex_group_name_to_vertex_index(name: str) -> Optional[int]:
    """Extracts the index of a vertex from a vertex group name if it was
    created earlier with `_vertex_index_to_vertex_group_name()`.
    """
    if (
        name.startswith("Skybrush[Vertex ")
        and name.endswith("]")
        and name[16:-1].isdigit()
    ):
        return int(name[16:-1])
    else:
        return None


def calculate_departure_index_of_drone(
    drone,
    drone_index: int,
    previous_entry: Optional[StoryboardEntry],
    previous_entry_index: int,
    previous_mapping: Optional[Mapping],
    targets_in_previous_formation: _LazyFormationTargetList,
) -> int:
    """Calculates the departure index of a drone (i.e. the index of the source
    marker that a drone is associated to) in a transition.
    """
    # In the departure sequence, the index of the drone is dictated
    # by the index of its associated marker / object within the
    # previous formation.
    if previous_mapping:
        previous_target_index = previous_mapping[drone_index]
        if previous_target_index is None:
            # drone did not participate in the previous formation
            return 0
        else:
            return previous_target_index

    # Previous mapping not known. All is not lost, however; we
    # can find which point in the previous formation the drone
    # must have belonged to by finding the constraint that binds
    # the drone to the previous storyboard entry

    if previous_entry is None:
        # This is the first entry. If we are calculating a
        # transition _into_ the first entry, the drones are
        # simply ordered according to how they are placed in the
        # Drones collection
        return drone_index

    previous_constraint = find_transition_constraint_between(
        drone=drone, storyboard_entry=previous_entry
    )
    if previous_constraint is None:
        # Either the drone did not participate in the previous formation,
        # or the previous storyboard entry is the first one in the
        # storyboard, in which case there are no constraints
        return drone_index if previous_entry_index == 0 else 0

    previous_obj = previous_constraint.target

    if previous_constraint.subtarget:
        # The constraint targets a vertex group of the object and the
        # subtarget property specifies which vertex to target within the
        # vertex group
        try:
            vertex_group = previous_obj.vertex_groups[previous_constraint.subtarget]
        except KeyError:
            # No such vertex group; something is wrong
            return 0

        # Blender's data structures are weird; we need to find the only vertex
        # in the vertex group, but we cannot get it directly fron the group.
        # The group only provides a weight() method that returns the weight of
        # a vertex if it is in the group, and throws an exception otherwise.
        #
        # We are going to use the subtarget name as a hint as it should contain
        # the vertex index if the user did not rename the subtarget. If this
        # fails, we will iterate over all vertices in the mesh to find the one
        # that is in the vertex group.
        vertex_index = _vertex_group_name_to_vertex_index(previous_constraint.subtarget)
        previous_mesh = cast(Mesh, previous_obj.data)
        if vertex_index is not None:
            previous_target = previous_mesh.vertices[vertex_index]
        else:
            for vertex in previous_mesh.vertices:
                try:
                    if vertex_group.weight(vertex.index) > 0:
                        previous_target = vertex
                        break
                except Exception:
                    pass
            else:
                # No vertex in the group; something is wrong
                return 0

    else:
        previous_target = previous_constraint.target

    return targets_in_previous_formation.find(previous_target)


def update_transition_constraint_properties(drone, entry: StoryboardEntry, marker, obj):
    """Updates the constraint that attaches a drone to its target in a
    transition.

    It is assumed (and not checked) that the storyboard entry is _not_ locked;
    in other words, it is assumed that we are allowed to modify the constraint
    corresponding to the transition.

    Parameters:
        drone: the drone to update
        entry: the storyboard entry; it will be used to select the appropriate
            constraint that corresponds to the drone _and_ the entry
        marker: the marker that the drone will be transitioning to; ``None``
            if the drone is not matched to a target in this transition
        obj: the parent mesh of the marker if the marker is a vertex in a
            Blender mesh, or the marker itself if the marker is a target
            mesh on its own (typically a Blender empty object)

    Returns:
        the Blender constraint that corresponds to the drone and the
        storyboard entry, or ``None`` if no such constraint is needed
        because the drone is unmatched
    """
    constraint = find_transition_constraint_between(drone=drone, storyboard_entry=entry)
    if marker is None:
        # This drone will not participate in this formation so
        # we need to delete the constraint that ties the drone
        # to the formation
        if constraint is not None:
            drone.constraints.remove(constraint)
        return None

    # If we don't have a constraint between the drone and the storyboard
    # entry, create one
    if constraint is None:
        constraint = create_transition_constraint_between(
            drone=drone, storyboard_entry=entry
        )
    else:
        # Make sure that the name of the constraint contains the
        # name of the formation even if the user renamed it
        set_constraint_name_from_storyboard_entry(constraint, entry)

    # Set the target of the constraint to the appropriate point of the
    # formation
    if marker is obj:
        # The marker itself is an object so it can be a constraint
        # target on its own
        constraint.target = marker
    else:
        # The marker is a vertex in a mesh so we need to create or
        # find a vertex group that contains the vertex only, and
        # use the vertex group as a subtarget
        index = marker.index
        vertex_group_name = _vertex_index_to_vertex_group_name(index)
        vertex_groups = obj.vertex_groups
        try:
            vertex_group = vertex_groups[vertex_group_name]
        except KeyError:
            # No such group, let's create it
            vertex_group = vertex_groups.new(name=vertex_group_name)

        # Ensure that the vertex group contains the target vertex
        # only in case the mesh was modified. Let's hope that
        # Blender is smart enough to make this a no-op if the vertex
        # group is okay as-is
        vertex_group.add([index], 1, "REPLACE")

        constraint.target = obj
        constraint.subtarget = vertex_group_name

    return constraint


def update_transition_constraint_influence(
    drone, constraint, descriptor: InfluenceCurveDescriptor
) -> None:
    """Updates the F-curve of the influence parameter of the constraint that
    attaches a given drone to the formation of a given storyboard entry.

    Parameters:
        drone: the drone that the transition constraint affects
        constraint: the transition constraint
        descriptor: the descriptor that describes how the influence F-curve
            should look like
    """
    # Construct the data path of the constraint we are going to
    # modify
    key = f"constraints[{constraint.name!r}].influence".replace("'", '"')

    # Create keyframes for the influence of the constraint
    ensure_action_exists_for_object(drone)

    # Apply the influence curve to the drone
    descriptor.apply(drone, key)


def update_transition_for_storyboard_entry(
    entry: StoryboardEntry,
    entry_index: int,
    drones,
    *,
    get_positions_of,
    previous_entry: Optional[StoryboardEntry],
    previous_mapping: Optional[Mapping],
    start_of_scene: int,
    start_of_next: Optional[int],
) -> Optional[Mapping]:
    """Updates the transition constraints corresponding to the given
    storyboard entry.

    Parameters:
        entry: the storyboard entry
        entry_index: index of the storyboard entry
        drones: the drones in the scene
        previous_entry: the storyboard entry that precedes the given entry;
            `None` if the given entry is the first one
        previous_mapping: the mapping from drone indices to target point
            indices in the _previous_ storyboard entry, if known; `None` if
            not known or if the given entry is the first one. Used for
            staggered transitions to determine when a given drone should
            depart from the previous formation
        start_of_scene: the first frame of the scene
        start_of_next: the frame where the _next_ storyboard entry starts;
            `None` if this is the last storyboard entry

    Returns:
        the mapping from drone index to marker index in the current
        formation, or `None` if the entry is a free segment or if the
        transition of the entry is locked and we are not allowed to touch it

    Raises:
        SkybrushStudioError: if an error happens while calculating transitions
    """
    if entry.is_locked:
        # entry is locked, nothing to do here
        return None

    formation = entry.formation
    if formation is None:
        # free segment, nothing to do here
        return None

    markers_and_objects = get_markers_and_related_objects_from_formation(formation)
    num_markers = len(markers_and_objects)
    end_of_previous = previous_entry.frame_end if previous_entry else start_of_scene

    # Calculate the positions to start the transition from. For most formations
    # this will be the current positions of the drones at the end of the previous
    # formation. However, the _first_ formation needs to be treated in a special
    # manner -- it has no preceding formation so we simply need to map each drone
    # to the marker with the same index, and we need to ensure that we have at
    # least as many markers as the number of drones
    if previous_entry:
        start_points = get_positions_of(drones, frame=end_of_previous)
    else:
        start_points = get_positions_of(
            (marker for marker, _ in markers_and_objects), frame=end_of_previous
        )
        if len(drones) != len(start_points):
            raise SkybrushStudioError(
                f"First formation has {len(start_points)} markers but the scene "
                f'contains {len(drones)} drones. Check the "Drones" collection '
                f"and the first formation for consistency."
            )

    mapping = calculate_mapping_for_transition_into_storyboard_entry(
        entry,
        start_points,
        num_targets=num_markers,
    )

    # Store mapping in Blender-compatible format for later use
    entry.update_mapping(mapping)

    recognized_point_mapping: Mapping = [None] * num_markers
    for drone_index, target_index in enumerate(mapping):
        if target_index is not None and 0 <= target_index < num_markers:
            recognized_point_mapping[target_index] = drone_index

    entry.update_recognized_point_mapping(recognized_point_mapping)

    # Calculate how many drones will participate in the transition
    num_drones_transitioning = sum(
        1 for target_index in mapping if target_index is not None
    )

    # Placeholder for the list of objects in the current and previous formations;
    # will be calculated on-demand for staggered transitions if needed
    objects_in_formation = _LazyFormationTargetList(entry)
    objects_in_previous_formation = _LazyFormationTargetList(previous_entry)

    # Create a mapping that maps indices of points in the source formation to
    # the corresponding schedule overrides (if any)
    schedule_override_map = entry.get_enabled_schedule_override_map()

    # Now we have the index of the target point that each drone
    # should be mapped to, and we have `None` for those drones that
    # will not participate in the formation
    todo: List[Callable[[], None]] = []
    for drone_index, drone in enumerate(drones):
        target_index = mapping[drone_index]
        if target_index is None:
            marker, obj = None, None
        else:
            marker, obj = markers_and_objects[target_index]

        constraint = update_transition_constraint_properties(drone, entry, marker, obj)

        if constraint is not None:
            # windup_start_frame can be later than end_of_previous for
            # staggered departures.

            windup_start_frame = end_of_previous
            start_frame = entry.frame_start
            departure_delay = 0
            arrival_delay = 0
            departure_index: Optional[int] = None

            if entry.is_staggered:
                # Determine the index of the drone in the departure sequence
                # and in the arrival sequence
                departure_index = calculate_departure_index_of_drone(
                    drone,
                    drone_index,
                    previous_entry,
                    entry_index - 1,
                    previous_mapping,
                    objects_in_previous_formation,
                )
                arrival_index = objects_in_formation.find(marker)

                departure_delay = entry.pre_delay_per_drone_in_frames * departure_index
                arrival_delay = -entry.post_delay_per_drone_in_frames * (
                    num_drones_transitioning - arrival_index - 1
                )

            if schedule_override_map:
                # Determine the index of the drone in the departure sequence
                # so we can look up whether there is an override for it. Note
                # that we do not need to do this again if we already have the
                # departure index
                if departure_index is None:
                    departure_index = calculate_departure_index_of_drone(
                        drone,
                        drone_index,
                        previous_entry,
                        entry_index - 1,
                        previous_mapping,
                        objects_in_previous_formation,
                    )

                override = schedule_override_map.get(departure_index)
                if override:
                    departure_delay = override.pre_delay
                    arrival_delay = -override.post_delay

            windup_start_frame += departure_delay
            start_frame += arrival_delay

            if previous_entry is None:
                # Special case: this is the constraint that holds the drones at
                # the first formation, so we need to set the start frame and
                # the windup start frame to the start of the scene
                start_frame = windup_start_frame = start_of_scene
            else:
                if windup_start_frame >= start_frame:
                    raise SkybrushStudioError(
                        f"Not enough time to plan staggered transition to "
                        f"formation {entry.name!r} at drone index {drone_index + 1} "
                        f"(1-based). Try decreasing departure or arrival delay "
                        f"or allow more time for the transition."
                    )

            # start_frame can be earlier than entry.frame_start for
            # staggered arrivals.
            descriptor = InfluenceCurveDescriptor(
                scene_start_frame=start_of_scene,
                windup_start_frame=windup_start_frame,
                start_frame=start_frame,
                end_frame=start_of_next,
            )

            # Do not update the influence curve now in case we have problems
            # with drones coming later in the enumeration; just store the
            # operation to call and then we'll do it in one batch at the end
            todo.append(
                partial(
                    update_transition_constraint_influence,
                    drone,
                    constraint,
                    descriptor,
                )
            )

    # Commit all the changes to the influence curves that we have planned above
    for func in todo:
        func()

    return mapping


@dataclass(frozen=True)
class SplitAllocationDescriptor:
    """Descriptor that captures split configuration for a branch."""

    branch_id: int
    num_drones: int


@dataclass(frozen=True)
class SplitAllocationRuntime:
    """Runtime descriptor of a split allocation with positional metadata."""

    branch_id: int
    count: int


@dataclass
class BranchState:
    """Runtime state of a storyboard branch during recalculation."""

    branch_id: int
    drones: List[Object]
    previous_entry: Optional[StoryboardEntry] = None
    previous_mapping: Optional[Mapping] = None


@dataclass
class SplitContext:
    """Represents an active split that must be merged later."""

    parent_state: BranchState
    allocations: Tuple[SplitAllocationRuntime, ...]


@dataclass
class RecalculationTask:
    """Descriptor for a single transition recalculation task to perform."""

    entry: StoryboardEntry
    entry_index: int
    branch_id: int = 0
    should_recalculate: bool = True
    is_split: bool = False
    is_merge: bool = False
    split_branches: Tuple[SplitAllocationDescriptor, ...] = ()
    previous_entry: Optional[StoryboardEntry] = None
    start_frame_of_next_entry: Optional[int] = None
    previous_mapping: Optional[Mapping] = None
    drones: Sequence[Object] = ()


def recalculate_transitions(
    tasks: Iterable[RecalculationTask], *, start_of_scene: int, drones: Sequence[Object]
) -> None:
    drone_list = list(drones)
    if not drone_list:
        return

    branch_states: dict[int, BranchState] = {
        0: BranchState(branch_id=0, drones=list(drone_list))
    }
    split_stack: List[SplitContext] = []

    with create_position_evaluator() as get_positions_of:
        for task in tasks:
            state = _select_branch_state(task, branch_states, split_stack, drone_list)

            task.previous_entry = state.previous_entry
            task.previous_mapping = state.previous_mapping
            task.drones = list(state.drones)

            mapping: Optional[Mapping] = None
            if task.should_recalculate and task.entry.formation:
                mapping = update_transition_for_storyboard_entry(
                    task.entry,
                    task.entry_index,
                    task.drones,
                    get_positions_of=get_positions_of,
                    previous_entry=task.previous_entry,
                    previous_mapping=task.previous_mapping,
                    start_of_scene=start_of_scene,
                    start_of_next=task.start_frame_of_next_entry,
                )
            elif task.entry.formation:
                mapping = _normalize_mapping(task.entry.get_mapping())

            state.previous_mapping = list(mapping) if mapping is not None else None
            if task.entry.formation:
                state.previous_entry = task.entry

            if task.is_split:
                if task.split_branches:
                    split_stack.append(
                        _apply_split_to_state(state, task, branch_states)
                    )
                else:
                    raise SkybrushStudioError(
                        f"Split entry {task.entry.name!r} has no branch allocations configured"
                    )
            # Merge handling is performed inside _select_branch_state

    for drone in drone_list:
        try:
            cleanup_actions_for_object(drone)
        except Exception:
            pass

    bpy.ops.skybrush.fix_constraint_ordering()
    invalidate_caches(clear_result=True)


def _select_branch_state(
    task: RecalculationTask,
    branch_states: dict[int, BranchState],
    split_stack: List[SplitContext],
    drones: Sequence[Object],
) -> BranchState:
    if task.is_merge:
        return _restore_parent_state(split_stack, branch_states, drones)

    state = branch_states.get(task.branch_id)
    if state is None:
        state = BranchState(branch_id=task.branch_id, drones=list(drones))
        branch_states[task.branch_id] = state
    return state


def _apply_split_to_state(
    parent_state: BranchState,
    task: RecalculationTask,
    branch_states: dict[int, BranchState],
) -> SplitContext:
    total_requested = sum(max(0, int(descriptor.num_drones)) for descriptor in task.split_branches)
    total_available = len(parent_state.drones)
    if total_requested != total_available:
        raise SkybrushStudioError(
            f"Split entry {task.entry.name!r} requests {total_requested} drones "
            f"but branch #{parent_state.branch_id} provides {total_available}"
        )

    mapping = parent_state.previous_mapping
    allocations: list[SplitAllocationRuntime] = []
    offset = 0
    seen: set[int] = set()
    for descriptor in task.split_branches:
        branch_id = int(descriptor.branch_id)
        if branch_id in seen:
            raise SkybrushStudioError(
                f"Split entry {task.entry.name!r} defines branch id {branch_id} multiple times"
            )
        seen.add(branch_id)

        count = max(0, int(descriptor.num_drones))
        subset_drones = parent_state.drones[offset : offset + count]
        subset_mapping = None
        if mapping is not None:
            subset_mapping = list(mapping[offset : offset + count])

        child_state = BranchState(
            branch_id=branch_id,
            drones=subset_drones,
            previous_entry=parent_state.previous_entry,
            previous_mapping=subset_mapping,
        )
        branch_states[branch_id] = child_state
        allocations.append(SplitAllocationRuntime(branch_id=branch_id, count=count))
        offset += count

    branch_states.pop(parent_state.branch_id, None)
    return SplitContext(parent_state=parent_state, allocations=tuple(allocations))


def _restore_parent_state(
    split_stack: List[SplitContext],
    branch_states: dict[int, BranchState],
    drones: Sequence[Object],
) -> BranchState:
    if not split_stack:
        state = branch_states.get(0)
        if state is None:
            state = BranchState(branch_id=0, drones=list(drones))
            branch_states[0] = state
        return state

    context = split_stack.pop()
    parent_state = context.parent_state
    merged_mapping: Optional[List[Optional[int]]] = []
    mapping_valid = True
    last_entry: Optional[StoryboardEntry] = parent_state.previous_entry

    for allocation in context.allocations:
        child_state = branch_states.pop(allocation.branch_id, None)
        if child_state is None:
            mapping_valid = False
            continue

        if mapping_valid:
            child_mapping = child_state.previous_mapping
            if child_mapping is None or len(child_mapping) != allocation.count:
                mapping_valid = False
            else:
                merged_mapping.extend(child_mapping)

        if child_state.previous_entry is not None:
            last_entry = child_state.previous_entry

    parent_state.previous_mapping = merged_mapping if mapping_valid else None
    parent_state.previous_entry = last_entry
    branch_states[parent_state.branch_id] = parent_state
    return parent_state


def _normalize_mapping(mapping: Optional[Mapping]) -> Optional[Mapping]:
    if not mapping:
        return None

    normalized: List[Optional[int]] = []
    for value in mapping:
        if value is None:
            normalized.append(None)
        else:
            ivalue = int(value)
            normalized.append(None if ivalue < 0 else ivalue)
    return normalized


class RecalculateTransitionsOperator(StoryboardOperator):
    """Recalculates all transitions in the show based on the current storyboard."""

    bl_idname = "skybrush.recalculate_transitions"
    bl_label = "Recalculate Transitions"
    bl_description = (
        "Recalculates all transitions in the show based on the current storyboard"
    )
    bl_options = {"UNDO"}

    scope = EnumProperty(
        items=[
            ("ALL", "Entire storyboard", "", "SEQUENCE", 1),
            ("CURRENT_FRAME", "Current frame", "", "EMPTY_SINGLE_ARROW", 2),
            None,
            (
                "TO_SELECTED",
                "To selected formation",
                "",
                "TRACKING_BACKWARDS_SINGLE",
                3,
            ),
            (
                "FROM_SELECTED",
                "From selected formation",
                "",
                "TRACKING_FORWARDS_SINGLE",
                4,
            ),
            (
                "FROM_SELECTED_TO_END",
                "From selected formation to end",
                "",
                "TRACKING_FORWARDS",
                5,
            ),
        ],
        name="Scope",
        description=(
            "Scope of the operator that defines which transitions must be recalculated"
        ),
        default="ALL",
    )

    only_with_valid_storyboard = True

    def execute_on_storyboard(self, storyboard: Storyboard, entries, context):
        # Get all the drones
        drones = Collections.find_drones().objects

        # If there are no drones, show a reasonable error message to the user
        if not drones:
            self.report({"ERROR"}, "You need to create some drones first")
            return {"CANCELLED"}

        tasks = self._get_transitions_to_process(storyboard, entries)
        if not any(task.should_recalculate for task in tasks):
            self.report({"INFO"}, "No transitions match the selected scope")
            return {"CANCELLED"}

        start_of_scene = min(context.scene.frame_start, storyboard.frame_start)

        try:
            with call_api_from_blender_operator(self, "transition planner"):
                recalculate_transitions(
                    tasks,
                    start_of_scene=start_of_scene,
                    drones=drones,
                )
            success = True
        except Exception:
            success = False

        return {"FINISHED"} if success else {"CANCELLED"}

    def _get_transitions_to_process(
        self, storyboard: Storyboard, entries: Sequence[StoryboardEntry]
    ) -> List[RecalculationTask]:
        """Collects recalculation tasks together with branch metadata."""

        tasks: List[RecalculationTask] = []
        active_index = int(storyboard.active_entry_index)
        num_entries = len(entries)

        if self.scope == "FROM_SELECTED":
            condition = (
                (active_index + 1).__eq__
                if active_index < num_entries - 1
                else constant(False)
            )
        elif self.scope == "TO_SELECTED":
            condition = active_index.__eq__
        elif self.scope == "FROM_SELECTED_TO_END":
            condition = active_index.__le__
        elif self.scope == "CURRENT_FRAME":
            frame = bpy.context.scene.frame_current
            index = storyboard.get_index_of_entry_after_frame(frame)
            condition = index.__eq__
        elif self.scope == "ALL":
            condition = constant(True)
        else:
            condition = constant(False)

        for index, entry in enumerate(entries):
            purpose = StoryboardEntryPurpose[entry.purpose]
            is_split = purpose == StoryboardEntryPurpose.SPLIT
            is_merge = purpose == StoryboardEntryPurpose.MERGE
            branch_id = int(getattr(entry, "split_id", 0))

            if is_split:
                split_branches = tuple(
                    SplitAllocationDescriptor(
                        branch_id=int(allocation.branch_id),
                        num_drones=int(allocation.num_drones),
                    )
                    for allocation in entry.split_allocations
                    if int(allocation.branch_id) > 0
                )
            else:
                split_branches = ()

            tasks.append(
                RecalculationTask(
                    entry=entry,
                    entry_index=index,
                    branch_id=branch_id,
                    should_recalculate=bool(condition(index)) and not entry.is_locked,
                    is_split=is_split,
                    is_merge=is_merge,
                    split_branches=split_branches,
                )
            )

        tasks_by_branch: dict[int, List[RecalculationTask]] = {}
        for task in tasks:
            tasks_by_branch.setdefault(task.branch_id, []).append(task)

        for branch_tasks in tasks_by_branch.values():
            branch_tasks.sort(
                key=lambda task: (task.entry.frame_start, task.entry.frame_end, task.entry_index)
            )
            for previous, current in zip(branch_tasks, branch_tasks[1:]):
                previous.start_frame_of_next_entry = current.entry.frame_start

        return tasks
