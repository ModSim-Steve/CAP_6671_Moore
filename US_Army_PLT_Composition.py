"""
US Army Platoon Composition

Purpose:

This file defines the structure and composition of a standard US Army infantry platoon
for use in a military simulation environment. It includes detailed representations of
regular infantry squads and a weapons squad, along with their equipment and formations.



General Composition:

1 Platoon consists of 3 basic squads and 1 weapons squad
Each basic squad has 1 Squad Leader (SL) and 2 teams (Alpha and Bravo)
Each team has 1 Team Leader (TL), 1 Grenadier (GRN), 1 Rifleman (RFLM), and 1 Automatic Rifleman (AR)
The weapons squad has 1 Squad Leader, 2 Gun Teams (2 soldiers each), and 2 Javelin Teams (2 soldiers each)
Leadership positions (PL, SL, TL) have their own agent for decision-making purposes



Equipment Details:

M4 Rifle (Primary weapon for most soldiers):

Range: 500 meters (50 spaces)
Ammo: 210 rounds (7 magazines of 30 rounds each)
Fire rate: 1 round per action
Effect: Precision (straight red line)


M249 Light Machine Gun (Primary weapon for AR):

Range: 800 meters (80 spaces)
Ammo: 600 rounds (3 magazines of 200 rounds each)
Fire rate: 6 rounds per action
Effect: Area weapon (red narrow transparent vector, 6 spaces wide at max range)


M320 Grenade Launcher (Secondary weapon for GRN):

Range: 350 meters (35 spaces)
Ammo: 12 rounds (loaded one at a time)
Fire rate: 1 round per action
Effect: Area damage (100 hp at target, 70 hp 1 space away, 30 hp 2 spaces away)


M240B Machine Gun (Primary weapon for Gun Teams in Weapons Squad):

Range: 1800 meters (180 spaces)
Ammo: 1000 rounds (5 magazines of 200 rounds each)
Fire rate: 9 rounds per action
Effect: Area weapon (similar to M249 but with greater range and fire rate)


Javelin (Secondary weapon for Javelin Teams in Weapons Squad):

Range: 2000 meters (200 spaces)
Ammo: 3 rounds
Fire rate: 1 round per action
Effect: Area damage (similar to M320 but with greater range)



Weapon Damage and Hit Probability:

Damage for weapons varies based on target distance (see in-code comments)
Hit probability for all weapons:

100% if target is ≤ 2/4 of max range
80% if target is between 2/4 and 3/4 of max range
70% if target is ≥ 3/4 of max range



Additional Features:

Formations: Includes functions to apply various formations (wedge, file, line) at team, squad, and platoon levels
Succession of Command: Implements logic for replacing incapacitated leaders at all levels
Health Status: Tracks health status of all soldiers (GREEN, AMBER, RED, BLACK)
Weapon Usage: Includes methods for firing weapons and calculating damage

Note: In the environment, 1 space equals 10 meters.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Union
from enum import Enum
import random
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class HealthStatus(Enum):
    GREEN = "green"
    AMBER = "amber"
    RED = "red"
    BLACK = "black"


class AnimationState(Enum):
    IDLE = "idle"
    MOVING = "moving"
    FIRING = "firing"
    RELOADING = "reloading"


class MovementTechnique(Enum):
    TRAVELING = "traveling"
    BOUNDING = "bounding"


class FormationType(Enum):
    WAGON_WHEEL = "wagon_wheel"  # Formations that rotate around pivot point
    FOLLOW_LEADER = "follow_leader"  # Column formations that follow path of leader


def get_formation_type(formation_name: str) -> FormationType:
    wagon_wheel_formations = [
        "team_wedge", "team_line", "gun_team", "javelin_team",
        "squad_line_team_wedge", "platoon_line_squad_line"
    ]
    return (FormationType.WAGON_WHEEL if formation_name in wagon_wheel_formations
            else FormationType.FOLLOW_LEADER)


@dataclass
class Weapon:
    name: str
    range: int  # in spaces (1 space = 10 meters)
    ammo_count: int
    fire_rate: int  # rounds per action
    firing_animation: str
    is_area_weapon: bool
    area_effect_width: int = 0  # Only applicable for area weapons

    def calculate_hit_probability(self, distance: int) -> float:
        range_ratio = distance / self.range
        if range_ratio <= 0.5:
            return 1.0
        elif 0.5 < range_ratio < 0.75:
            return 0.8
        else:
            return 0.7

    def calculate_damage(self, distance: int) -> int:
        range_ratio = distance / self.range
        if range_ratio >= 0.75:
            return 80 if random.random() < 0.1 else 30
        elif 0.5 <= range_ratio < 0.75:
            return 80 if random.random() < 0.1 else 50
        else:
            return 90 if random.random() < 0.1 else 80


@dataclass
class GrenadeLauncher(Weapon):
    def calculate_damage(self, distance: int) -> Tuple[int, int, int]:
        return 100, 70, 30  # Damage at target, 1 space away, and 2 spaces away


@dataclass
class AreaDamageWeapon(Weapon):
    def calculate_damage(self, distance: int) -> Tuple[int, int, int]:
        return 100, 70, 30  # Damage at target, 1 space away, and 2 spaces away


@dataclass
class Soldier:
    role: str
    health: int
    max_health: int
    primary_weapon: Weapon
    secondary_weapon: Optional[Weapon]
    observation_range: int
    engagement_range: int
    position: Tuple[int, int]
    is_leader: bool
    animation_state: AnimationState = AnimationState.IDLE

    @property
    def is_alive(self):
        return self.health > 0

    @property
    def health_status(self) -> HealthStatus:
        health_percentage = self.health / self.max_health
        if health_percentage > 0.8:
            return HealthStatus.GREEN
        elif health_percentage > 0.6:
            return HealthStatus.AMBER
        elif health_percentage > 0:
            return HealthStatus.RED
        else:
            return HealthStatus.BLACK

    def update_animation(self, new_state: AnimationState):
        self.animation_state = new_state

    def move_to(self, new_position: Tuple[int, int]):
        self.position = new_position
        self.update_animation(AnimationState.MOVING)

    def fire_weapon(self, is_primary: bool = True):
        weapon = self.primary_weapon if is_primary else self.secondary_weapon
        if weapon and weapon.ammo_count >= weapon.fire_rate:
            weapon.ammo_count -= weapon.fire_rate
            self.update_animation(AnimationState.FIRING)
            return True
        return False


@dataclass
class TeamMember:
    soldier: Soldier
    team_id: str


@dataclass
class TeamDebug:
    """Debug settings and functions for Team movement validation."""
    enabled: bool = False

    def validate_formation(self, team: 'Team', frame_num: int, total_frames: int, phase: str):
        """Main validation function that checks both spacing and rotation."""
        if not self.enabled or phase.startswith("Pre-") or phase.startswith("Post-"):
            return

        print(f"\n{'=' * 20} Frame {frame_num}/{total_frames} {'=' * 20}")
        print(f"Phase: {phase}")

        # Print basic team info
        print(f"\nTeam {team.team_id}:")
        print(f"Formation: {team.current_formation}")
        print(f"Orientation: {team.orientation}°")

        # Validate spacing
        self._validate_spacing(team)

        # Validate rotation
        self._validate_rotation(team)

    def _validate_spacing(self, team: 'Team'):
        """Validate spacing between team members."""
        if not self.enabled:
            return

        print("\nSpacing Validation:")
        tl_pos = team.leader.soldier.position

        for member in team.members:
            # Calculate actual spacing
            rel_x = member.soldier.position[0] - tl_pos[0]
            rel_y = member.soldier.position[1] - tl_pos[1]
            actual_spacing = int(math.sqrt(rel_x * rel_x + rel_y * rel_y))

            # Get template spacing
            template_pos = team.formation_positions.get(member.soldier.role, (0, 0))
            template_x, template_y = template_pos
            template_spacing = int(math.sqrt(template_x * template_x + template_y * template_y))

            # Print validation info
            print(f"\n{member.soldier.role}:")
            print(f"  Template spacing: {template_spacing}")
            print(f"  Actual spacing: {actual_spacing}")
            print(f"  Template position: ({template_x}, {template_y})")
            print(f"  Actual relative position: ({rel_x}, {rel_y})")

            # Check for spacing issues
            if abs(template_spacing - actual_spacing) > 1:  # 1 unit tolerance
                print(f"  WARNING: Spacing mismatch! Difference: {abs(template_spacing - actual_spacing)}")

    def _validate_rotation(self, team: 'Team'):
        """Validate rotation and positions after movement."""
        if not self.enabled:
            return

        print("\nRotation Validation:")
        tl_pos = team.leader.soldier.position
        print(f"Team orientation: {team.orientation}°")

        # Convert team orientation to radians for position calculations
        orientation_rad = math.radians(team.orientation)

        for member in team.members:
            template_pos = team.formation_positions.get(member.soldier.role, (0, 0))

            # Calculate expected position after rotation
            expected_x = int(template_pos[0] * math.cos(orientation_rad) -
                             template_pos[1] * math.sin(orientation_rad))
            expected_y = int(template_pos[0] * math.sin(orientation_rad) +
                             template_pos[1] * math.cos(orientation_rad))

            # Get actual relative position
            actual_x = member.soldier.position[0] - tl_pos[0]
            actual_y = member.soldier.position[1] - tl_pos[1]

            print(f"\n{member.soldier.role}:")
            print(f"  Expected relative position: ({expected_x}, {expected_y})")
            print(f"  Actual relative position: ({actual_x}, {actual_y})")

            # Check for position mismatches
            if abs(expected_x - actual_x) > 1 or abs(expected_y - actual_y) > 1:
                print(
                    f"  WARNING: Position mismatch! Difference: ({abs(expected_x - actual_x)}, {abs(expected_y - actual_y)})")


@dataclass
class Team:
    team_id: str
    leader: TeamMember
    members: List[TeamMember] = field(default_factory=list)
    orientation: int = 0  # Orientation in degrees, 0 is North
    formation: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_formation: str = "team_wedge"  # Formation name
    formation_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # Formation positions
    parent_unit: Optional['Squad'] = None
    debug: TeamDebug = field(default_factory=lambda: TeamDebug(enabled=False))

    @property
    def all_members(self):
        return [self.leader] + self.members

    @property
    def alive_members(self):
        return [member for member in self.all_members if member.soldier.is_alive]

    def add_member(self, role: str, primary_weapon: Weapon, secondary_weapon: Optional[Weapon],
                   observation_range: int, engagement_range: int, position: Tuple[int, int]):
        """
        Add a new member to the team.
        For regular teams, roles should be one of: Team Leader, Automatic Rifleman,
        Grenadier, or Rifleman
        """
        soldier = Soldier(
            role=role,
            health=100,
            max_health=100,
            primary_weapon=primary_weapon,
            secondary_weapon=secondary_weapon,
            observation_range=observation_range,
            engagement_range=engagement_range,
            position=position,
            is_leader=(role == "Team Leader")
        )
        member = TeamMember(soldier, self.team_id)

        if role == "Team Leader":
            if self.leader is not None:
                raise ValueError(f"Team {self.team_id} already has a leader")
            self.leader = member
        else:
            self.members.append(member)

    def check_and_replace_leader(self):
        if not self.leader.soldier.is_alive:
            alive_members = [m for m in self.members if m.soldier.is_alive]
            if alive_members:
                new_leader = alive_members[0]  # Select the first alive member as new leader
                new_leader.soldier.is_leader = True
                self.leader = new_leader
                self.members.remove(new_leader)
                print(f"New leader for {self.team_id}: {new_leader.soldier.role}")
            else:
                print(f"All members of {self.team_id} are incapacitated.")

    def set_formation(self, formation_positions: Dict[str, Tuple[int, int]],
                      formation_name: str):
        """Set team formation positions and name."""
        self.formation_positions = formation_positions
        self.current_formation = formation_name
        self.apply_formation(self.leader.soldier.position)

    def apply_formation(self, base_position: Tuple[int, int] = (0, 0)):
        """Apply current formation from the given base position."""
        base_x, base_y = base_position
        # Position leader
        leader_rel_x, leader_rel_y = self.formation_positions.get("Team Leader", (0, 0))
        self.leader.soldier.position = (base_x + leader_rel_x, base_y + leader_rel_y)

        # Position other members
        for member in self.members:
            rel_x, rel_y = self.formation_positions.get(member.soldier.role, (0, 0))
            member.soldier.position = (base_x + rel_x, base_y + rel_y)

    def execute_movement(self, direction: Tuple[int, int], distance: int,
                         technique: MovementTechnique = MovementTechnique.TRAVELING) -> List[Dict]:
        """Primary movement executor that chooses appropriate movement type."""
        frames = []
        steps = 10

        if self.debug.enabled:
            self.debug.validate_formation(self, 1, steps + 1, "Starting Movement Execution")

        formation_type = get_formation_type(self.current_formation)
        if formation_type == FormationType.WAGON_WHEEL:
            frames = self._execute_wagon_wheel(direction, distance)
        else:  # FormationType.FOLLOW_LEADER
            frames = self._execute_follow_leader_movement(direction, distance)

        if self.debug.enabled:
            self.debug.validate_formation(self, steps + 1, steps + 1, "Movement Execution Complete")

        return frames

    def _execute_wagon_wheel(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute wagon wheel movement with rotation and translation."""
        frames = []
        steps = 10
        rotation_steps = 4

        if self.debug.enabled:
            self.debug.validate_formation(self, 1, rotation_steps + steps,
                                          "Starting Wagon Wheel Movement")

        # Calculate rotation needed to face movement direction
        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        # Calculate new orientation - change to handle all cases identically
        target_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        rotation_needed = ((target_orientation - self.orientation + 180) % 360) - 180

        if self.debug.enabled:
            print(f"\nCurrent orientation: {self.orientation}°")
            print(f"Target orientation: {target_orientation}°")
            print(f"Rotation needed: {rotation_needed}°")

        # Always rotate if there's any direction change
        if dx != 0 or dy != 0:
            rotation_per_step = rotation_needed // rotation_steps

            for step in range(rotation_steps):
                self._move_wagon_wheel((0, 0), 0, rotation_per_step)
                frames.append(self._capture_current_positions())

                if self.debug.enabled:
                    print(f"Rotation step {step + 1}: orientation now {self.orientation}°")

        # Movement phase
        magnitude = int(math.sqrt(dx * dx + dy * dy))
        step_distance = distance // steps

        for step in range(steps):
            self._move_wagon_wheel(direction, step_distance)
            frames.append(self._capture_current_positions())

        return frames

    def _move_wagon_wheel(self, direction: Tuple[int, int], distance: int,
                          rotation_amount: int = 0) -> None:
        """Execute single wagon wheel movement step."""
        if self.debug.enabled:
            self.debug.validate_formation(self, 1, 1, "Pre-Movement Validation")

        # Store leader's initial position
        leader_x, leader_y = self.leader.soldier.position

        # Always apply rotation first
        if rotation_amount != 0:
            # Update orientation first
            self.orientation = (self.orientation + rotation_amount) % 360
            angle_rad = math.radians(self.orientation - 90)  # Adjust to face movement direction

            # Rotate each member around leader using template positions
            for member in self.members:
                # Use template positions
                template_x, template_y = self.formation_positions[member.soldier.role]

                # Calculate rotated position
                rot_x = int(template_x * math.cos(angle_rad) - template_y * math.sin(angle_rad))
                rot_y = int(template_x * math.sin(angle_rad) + template_y * math.cos(angle_rad))

                # Apply rotated position relative to leader
                member.soldier.position = (leader_x + rot_x, leader_y + rot_y)

        # Then apply movement
        if distance > 0:
            dx, dy = direction
            magnitude = int(math.sqrt(dx * dx + dy * dy))
            if magnitude > 0:
                move_dx = (dx * distance) // magnitude
                move_dy = (dy * distance) // magnitude

                # Move entire formation
                self.leader.soldier.position = (leader_x + move_dx, leader_y + move_dy)
                for member in self.members:
                    member.soldier.position = (
                        member.soldier.position[0] + move_dx,
                        member.soldier.position[1] + move_dy
                    )

        if self.debug.enabled:
            self.debug.validate_formation(self, 1, 1, "Post-Movement Validation")

    def _execute_follow_leader_movement(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute follow-the-leader movement with proper path following."""
        frames = []
        steps = 10

        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        # Calculate new orientation for movement direction
        new_orientation = int((math.degrees(math.atan2(dy, dx))) % 360)
        rotation_needed = ((new_orientation - self.orientation + 180) % 360) - 180

        if self.debug.enabled:
            print(f"\nStarting follow leader movement:")
            print(f"Current orientation: {self.orientation}°")
            print(f"New orientation: {new_orientation}°")
            print(f"Rotation needed: {rotation_needed}°")

        # Store initial positions for path history
        current_pos = self.leader.soldier.position
        path_history = [current_pos]

        # Store member spacing from leader
        spacing = []
        for member in self.members:
            spacing.append(member.soldier.position[1] - current_pos[1])  # For column formation

        # First rotate at current position if needed
        if abs(rotation_needed) > 1:
            rotation_per_step = rotation_needed // 4
            for _ in range(4):
                self.orientation = (self.orientation + rotation_per_step) % 360
                frames.append(self._capture_current_positions())

        # Calculate movement increments
        magnitude = int(math.sqrt(dx * dx + dy * dy))
        step_distance = distance // steps
        move_dx = (dx * step_distance) // magnitude
        move_dy = (dy * step_distance) // magnitude

        # Execute movement
        for step in range(steps):
            # Move leader
            current_pos = self.leader.soldier.position
            new_x = current_pos[0] + move_dx
            new_y = current_pos[1] + move_dy
            self.leader.soldier.position = (new_x, new_y)
            path_history.append((new_x, new_y))

            # Move each follower to leader's previous positions
            for i, member in enumerate(self.members):
                if len(path_history) > i + 1:  # Make sure we have enough path history
                    member.soldier.position = path_history[
                        -(i + 2)]  # Use second-to-last position for first follower, etc.

            frames.append(self._capture_current_positions())

            if self.debug.enabled:
                print(f"\nStep {step + 1}:")
                print(f"Leader position: {self.leader.soldier.position}")
                positions = [m.soldier.position for m in self.members]
                print(f"Member positions: {positions}")

        return frames

    def _capture_current_positions(self) -> Dict:
        """Capture current positions of all team members for animation."""
        return {
            'unit_type': 'Team',
            'team_id': self.team_id,
            'positions': [
                             {
                                 'role': self.leader.soldier.role,
                                 'position': self.leader.soldier.position,
                                 'is_leader': True
                             }
                         ] + [
                             {
                                 'role': member.soldier.role,
                                 'position': member.soldier.position,
                                 'is_leader': False
                             } for member in self.members
                         ]
        }


@dataclass
class SquadDebug:
    """Debug settings and functions for Squad movement validation."""
    enabled: bool = False
    path_history: List[Tuple[int, int]] = field(default_factory=list)
    pivot_points: List[Tuple[int, int]] = field(default_factory=list)
    element_orientations: Dict[str, List[int]] = field(default_factory=lambda: {
        "Squad": [],
        "Alpha Team": [],
        "Bravo Team": []
    })
    formation_snapshots: List[Dict] = field(default_factory=list)

    def validate_formation(self, squad: 'Squad', frame_num: int, total_frames: int, phase: str):
        """Main validation function that checks squad and team formations."""
        if not self.enabled:
            return

        print(f"\n{'=' * 20} Frame {frame_num}/{total_frames} {'=' * 20}")
        print(f"Phase: {phase}")

        # Create formation snapshot
        snapshot = {
            'frame': frame_num,
            'phase': phase,
            'squad_orientation': squad.orientation,
            'sl_position': squad.leader.position,
            'teams': {}
        }

        # Record element orientations
        self.element_orientations["Squad"].append(squad.orientation)
        self.element_orientations["Alpha Team"].append(squad.alpha_team.orientation)
        self.element_orientations["Bravo Team"].append(squad.bravo_team.orientation)

        # Print squad info
        print(f"\nSQUAD LEVEL INFORMATION:")
        print(f"Formation: {squad.current_formation}")
        print(f"Squad Leader Position: {squad.leader.position}")
        print(f"Squad Orientation: {squad.orientation}°")

        # Validate team formations and positions
        self._validate_team_formations(squad, snapshot)
        self._validate_squad_spacing(squad, snapshot)

        # Store snapshot
        self.formation_snapshots.append(snapshot)

        # Print path information if in follow leader mode
        if "Movement Step" in phase:
            self._print_path_info()

    def validate_follow_leader(self, element_name: str, position: Tuple[int, int],
                               pivot_point: Optional[Tuple[int, int]], has_rotated: bool):
        """Validate specific follow leader movement aspects."""
        if not self.enabled:
            return

        print(f"\n{element_name} Status:")
        print(f"Current Position: {position}")

        if pivot_point:
            dist_to_pivot = math.sqrt((position[0] - pivot_point[0]) ** 2 +
                                      (position[1] - pivot_point[1]) ** 2)
            print(f"Distance to pivot point: {dist_to_pivot:.2f}")
            print(f"Has rotated: {has_rotated}")

            # Store pivot point if new
            if pivot_point not in self.pivot_points:
                self.pivot_points.append(pivot_point)

        if element_name == "Lead Team":
            self.path_history.append(position)

    def log_movement_start(self, direction: Tuple[int, int], distance: int, steps: int):
        """Log movement initialization details."""
        if not self.enabled:
            return

        print("\n=== Starting Squad Follow Leader Movement ===")
        print(f"Direction vector: ({direction[0]}, {direction[1]})")
        print(f"Total distance: {distance}")
        print(f"Steps: {steps}")

    def log_formation_setup(self, squad: 'Squad', pre_post: str):
        """Log formation setup details."""
        if not self.enabled:
            return

        print(f"\n=== {pre_post} Formation Setup ===")
        print(f"Squad orientation: {squad.orientation}°")
        print(f"SL position: {squad.leader.position}")
        print(f"Alpha team position: {squad.alpha_team.leader.soldier.position}")
        print(f"Bravo team position: {squad.bravo_team.leader.soldier.position}")

    def log_step_start(self, step: int, total_steps: int):
        """Log beginning of movement step."""
        if not self.enabled:
            return

        print(f"\n=== Step {step + 1}/{total_steps} ===")

    def log_team_rotation(self, team: Team, is_start: bool):
        """Log team rotation details."""
        if not self.enabled:
            return

        status = "Starting" if is_start else "Completed"
        print(f"\n{team.team_id} Team Rotation - {status}:")
        print(f"Orientation: {team.orientation}°")
        self._validate_team_formations_single(team)

    def log_team_movement(self, team: Team, old_pos: Tuple[int, int], new_pos: Tuple[int, int]):
        """Log team movement details."""
        if not self.enabled:
            return

        print(f"\n{team.team_id} Team Movement:")
        print(f"Old position: {old_pos}")
        print(f"New position: {new_pos}")
        self._validate_team_formations_single(team)

    def log_trail_team_status(self, team: Team, pivot_point: Tuple[int, int], reached_pivot: bool):
        """Log trail team status relative to pivot point."""
        if not self.enabled:
            return

        dist_to_pivot = self._get_spacing(team.leader.soldier.position, pivot_point)
        print("\nTrail Team Status:")
        print(f"Current position: {team.leader.soldier.position}")
        print(f"Distance to pivot: {dist_to_pivot:.2f}")
        print(f"Current orientation: {team.orientation}°")
        print(f"Pivot reached: {reached_pivot}")

    def _validate_team_formations(self, squad: 'Squad', snapshot: Dict):
        """Validate formations of both teams."""
        print("\nTEAM FORMATIONS:")
        for team in [squad.alpha_team, squad.bravo_team]:
            print(f"\n{team.team_id} TEAM:")
            tl_pos = team.leader.soldier.position
            print(f"Team Leader Position: {tl_pos}")
            print(f"Team Orientation: {team.orientation}°")

            # Store team info in snapshot
            team_info = {
                'orientation': team.orientation,
                'tl_position': tl_pos,
                'members': []
            }

            # Print team member positions
            print("Team Members:")
            for member in team.members:
                rel_x = member.soldier.position[0] - tl_pos[0]
                rel_y = member.soldier.position[1] - tl_pos[1]

                # Get template position for comparison
                template_pos = team.formation_positions[member.soldier.role]
                template_orientation = math.radians(team.orientation - 90)
                expected_x = int(template_pos[0] * math.cos(template_orientation) -
                                 template_pos[1] * math.sin(template_orientation))
                expected_y = int(template_pos[0] * math.sin(template_orientation) +
                                 template_pos[1] * math.cos(template_orientation))

                role_abbrev = {
                    "Automatic Rifleman": "AR",
                    "Grenadier": "GRN",
                    "Rifleman": "RFLM"
                }.get(member.soldier.role, member.soldier.role)

                print(f"  * {role_abbrev}: {member.soldier.position}")
                print(f"    Relative to TL: actual ({rel_x}, {rel_y}), expected ({expected_x}, {expected_y})")

                # Check for position discrepancy
                if abs(rel_x - expected_x) > 1 or abs(rel_y - expected_y) > 1:
                    print(f"    WARNING: Position mismatch! Difference: "
                          f"({abs(rel_x - expected_x)}, {abs(rel_y - expected_y)})")

                # Store member info in snapshot
                team_info['members'].append({
                    'role': role_abbrev,
                    'position': member.soldier.position,
                    'relative_pos': (rel_x, rel_y),
                    'expected_pos': (expected_x, expected_y)
                })

            snapshot['teams'][team.team_id] = team_info

    def _validate_team_formations_single(self, team: Team):
        """Validate formation of a single team."""
        print(f"\n{team.team_id} Team:")
        tl_pos = team.leader.soldier.position
        print(f"Team Leader Position: {tl_pos}")
        print(f"Team Orientation: {team.orientation}°")

        print("Team Members:")
        for member in team.members:
            rel_x = member.soldier.position[0] - tl_pos[0]
            rel_y = member.soldier.position[1] - tl_pos[1]
            role_abbrev = {
                "Automatic Rifleman": "AR",
                "Grenadier": "GRN",
                "Rifleman": "RFLM"
            }.get(member.soldier.role, member.soldier.role)
            print(f"  * {role_abbrev}: {member.soldier.position} [relative to TL: ({rel_x}, {rel_y})]")

    def _validate_squad_spacing(self, squad: 'Squad', snapshot: Dict):
        """Validate spacing between squad leader and teams."""
        print("\nSQUAD SPACING VALIDATION:")
        sl_pos = squad.leader.position
        snapshot['spacing'] = {}

        # Calculate and validate team spacing
        for team in [squad.alpha_team, squad.bravo_team]:
            tl_pos = team.leader.soldier.position
            actual_distance = math.sqrt(
                (tl_pos[0] - sl_pos[0]) ** 2 +
                (tl_pos[1] - sl_pos[1]) ** 2
            )
            expected_distance = math.sqrt(
                sum(x * x for x in squad.formation_positions.get(f"{team.team_id} Team", (0, 0)))
            )
            print(f"{team.team_id} Team to SL: {actual_distance:.2f} [Expected: {expected_distance:.2f}]")

            snapshot['spacing'][team.team_id] = {
                'actual': actual_distance,
                'expected': expected_distance
            }

            if abs(actual_distance - expected_distance) > 1:  # 1 unit tolerance
                print(f"  WARNING: {team.team_id} Team spacing mismatch! "
                      f"Difference: {abs(actual_distance - expected_distance):.2f}")

    def _get_spacing(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate spacing between two positions."""
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def _print_path_info(self):
        """Print information about the movement path and pivot points."""
        print("\nPATH INFORMATION:")
        print(f"Total path points recorded: {len(self.path_history)}")
        print(f"Pivot points: {self.pivot_points}")

        print("\nElement Orientation History:")
        for element, orientations in self.element_orientations.items():
            if orientations:
                print(
                    f"{element}: {orientations[-1]}° (Previous: {orientations[-2] if len(orientations) > 1 else 'N/A'}°)")

    def get_movement_summary(self) -> Dict:
        """Generate summary of the movement execution."""
        return {
            'total_frames': len(self.formation_snapshots),
            'pivot_points': self.pivot_points,
            'path_length': len(self.path_history),
            'orientation_changes': {
                element: [abs(orientations[i] - orientations[i - 1])
                          for i in range(1, len(orientations))]
                for element, orientations in self.element_orientations.items()
            }
        }

    def clear_path_data(self):
        """Clear path-specific debug data for new movement."""
        self.path_history.clear()
        self.pivot_points.clear()
        self.element_orientations = {
            "Squad": [],
            "Alpha Team": [],
            "Bravo Team": []
        }
        self.formation_snapshots.clear()


@dataclass
class Squad:
    squad_id: str
    leader: Soldier
    alpha_team: Team
    bravo_team: Team
    orientation: int = 0
    formation: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_formation: str = "squad_column_team_wedge"  # Formation name
    formation_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # Formation positions
    parent_unit: Optional['Platoon'] = None
    debug: SquadDebug = field(default_factory=lambda: SquadDebug(enabled=False))

    @property
    def teams(self) -> List[Team]:
        """Provide consistent access to teams as a list."""
        return [self.alpha_team, self.bravo_team]

    @property
    def all_members(self):
        return [self.leader] + self.alpha_team.all_members + self.bravo_team.all_members

    @property
    def alive_members(self):
        return [member for member in self.all_members if member.is_alive]

    def check_and_replace_leader(self):
        if not self.leader.is_alive:
            for team in self.teams:
                if team.leader.soldier.is_alive:
                    self.leader = team.leader.soldier
                    self.leader.is_leader = True
                    print(f"New squad leader for {self.squad_id}: {team.team_id} leader")
                    team.check_and_replace_leader()  # Replace the team leader
                    return
            print(f"No available leaders for {self.squad_id}")

    def set_formation(self,
                      formation_positions: Dict[str, Tuple[int, int]],
                      team_formation_alpha: Dict[str, Tuple[int, int]],
                      team_formation_bravo: Dict[str, Tuple[int, int]],
                      formation_name: str):
        """Set squad and team formation positions and names."""
        self.formation_positions = formation_positions
        self.current_formation = formation_name
        self.alpha_team.set_formation(team_formation_alpha, "team_wedge_left")
        self.bravo_team.set_formation(team_formation_bravo, "team_wedge_right")

    def apply_formation(self, base_position: Tuple[int, int] = (0, 0)):
        """Apply current formation from the given base position."""
        base_x, base_y = base_position
        sl_x, sl_y = self.formation_positions.get("Squad Leader", (0, 0))
        self.leader.position = (base_x + sl_x, base_y + sl_y)

        alpha_x, alpha_y = self.formation_positions.get("Alpha Team", (0, 0))
        self.alpha_team.apply_formation((base_x + alpha_x, base_y + alpha_y))

        bravo_x, bravo_y = self.formation_positions.get("Bravo Team", (0, 0))
        self.bravo_team.apply_formation((base_x + bravo_x, base_y + bravo_y))

    def execute_movement(self, direction: Tuple[int, int], distance: int,
                         technique: MovementTechnique = MovementTechnique.TRAVELING) -> List[Dict]:
        """
        Execute movement using specified technique.
        """
        print(f"\n=== Starting Squad Movement ===")
        print(f"Movement technique: {technique.value}")
        frames = []

        if technique == MovementTechnique.TRAVELING:
            formation_type = get_formation_type(self.current_formation)
            if formation_type == FormationType.WAGON_WHEEL:
                frames = self._execute_wagon_wheel(direction, distance)
            else:
                frames = self._execute_follow_leader(direction, distance)
        else:  # BOUNDING
            frames = self._execute_bounding(direction, distance)

        return frames

    def _execute_wagon_wheel(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute wagon wheel movement with synchronized team rotations."""
        frames = []
        steps = 10
        rotation_steps = 4

        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        # Calculate rotation needed
        target_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        rotation_needed = ((target_orientation - self.orientation + 180) % 360) - 180

        if self.debug.enabled:
            self.debug.validate_formation(self, 1, steps + rotation_steps, "Initial Position")

        # Store squad leader's position as pivot point
        sl_pos = self.leader.position

        # Rotation phase
        if abs(rotation_needed) > 1:
            rotation_per_step = rotation_needed // rotation_steps

            for step in range(rotation_steps):
                # Update squad orientation
                self.orientation = (self.orientation + rotation_per_step) % 360
                current_angle_rad = math.radians(self.orientation - 90)

                # Rotate both teams around SL
                for team in [self.alpha_team, self.bravo_team]:
                    # Get base position for this team
                    base_pos = self.formation_positions.get(f"{team.team_id} Team", (0, 0))

                    # Rotate base position around SL
                    rot_x = int(base_pos[0] * math.cos(current_angle_rad) -
                                base_pos[1] * math.sin(current_angle_rad))
                    rot_y = int(base_pos[0] * math.sin(current_angle_rad) +
                                base_pos[1] * math.cos(current_angle_rad))

                    # Set new team leader position relative to SL
                    new_tl_pos = (sl_pos[0] + rot_x, sl_pos[1] + rot_y)
                    team.leader.soldier.position = new_tl_pos

                    # Update team orientation
                    team.orientation = self.orientation

                    # Rotate each team member around their TL
                    for member in team.members:
                        # Get template position for this member
                        template_pos = team.formation_positions[member.soldier.role]

                        # Rotate template position based on new orientation
                        member_angle_rad = math.radians(team.orientation - 90)
                        member_rot_x = int(template_pos[0] * math.cos(member_angle_rad) -
                                           template_pos[1] * math.sin(member_angle_rad))
                        member_rot_y = int(template_pos[0] * math.sin(member_angle_rad) +
                                           template_pos[1] * math.cos(member_angle_rad))

                        # Set member position relative to TL
                        member.soldier.position = (
                            new_tl_pos[0] + member_rot_x,
                            new_tl_pos[1] + member_rot_y
                        )

                frames.append(self._capture_current_positions())

                if self.debug.enabled:
                    self.debug.validate_formation(self, step + 2, steps + rotation_steps,
                                                  f"Rotation Step {step + 1}")

        # Movement phase
        magnitude = int(math.sqrt(dx * dx + dy * dy))
        step_distance = distance // steps
        move_dx = (dx * step_distance) // magnitude
        move_dy = (dy * step_distance) // magnitude

        for step in range(steps):
            if self.debug.enabled:
                self.debug.validate_formation(self, step + rotation_steps + 1,
                                              steps + rotation_steps,
                                              f"Movement Step {step + 1}")

            # Move squad leader
            new_sl_pos = (self.leader.position[0] + move_dx,
                          self.leader.position[1] + move_dy)
            self.leader.position = new_sl_pos

            # Move teams maintaining formation
            for team in [self.alpha_team, self.bravo_team]:
                base_pos = self.formation_positions.get(f"{team.team_id} Team", (0, 0))

                # Calculate new TL position relative to SL
                rot_x = int(base_pos[0] * math.cos(math.radians(self.orientation - 90)) -
                            base_pos[1] * math.sin(math.radians(self.orientation - 90)))
                rot_y = int(base_pos[0] * math.sin(math.radians(self.orientation - 90)) +
                            base_pos[1] * math.cos(math.radians(self.orientation - 90)))

                new_tl_pos = (new_sl_pos[0] + rot_x, new_sl_pos[1] + rot_y)
                team.leader.soldier.position = new_tl_pos

                # Move each team member maintaining their rotated positions relative to TL
                for member in team.members:
                    template_pos = team.formation_positions[member.soldier.role]

                    # Keep formation rotation during movement
                    member_angle_rad = math.radians(team.orientation - 90)
                    member_rot_x = int(template_pos[0] * math.cos(member_angle_rad) -
                                       template_pos[1] * math.sin(member_angle_rad))
                    member_rot_y = int(template_pos[0] * math.sin(member_angle_rad) +
                                       template_pos[1] * math.cos(member_angle_rad))

                    member.soldier.position = (
                        new_tl_pos[0] + member_rot_x,
                        new_tl_pos[1] + member_rot_y
                    )

            frames.append(self._capture_current_positions())

        return frames

    def _execute_follow_leader(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """
        Execute follow leader movement with proper element rotations at pivot points.
        Aligned with original implementation and updated helper functions.
        """
        frames = []
        steps = 10

        # Initialize debugging if enabled
        if self.debug.enabled:
            self.debug.clear_path_data()
            self.debug.log_movement_start(direction, distance, steps)
            self.debug.log_formation_setup(self, "Initial")

        frames.append(self._capture_current_positions())

        # Determine lead and trail teams
        lead_team = self.alpha_team
        trail_team = self.bravo_team

        # Calculate movement and rotation
        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        magnitude = int(math.sqrt(dx * dx + dy * dy))
        move_dx = (dx * distance // steps) // magnitude
        move_dy = (dy * distance // steps) // magnitude

        original_orientation = lead_team.orientation
        new_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        rotation_needed = ((new_orientation - original_orientation + 180) % 360) - 180

        # Store pivot point and initialize tracking
        pivot_point = lead_team.leader.soldier.position
        path_points = [pivot_point]

        if self.debug.enabled:
            self.debug.validate_follow_leader("Lead Team", pivot_point, None, False)
            print(f"\nMovement and rotation calculations:")
            print(f"Direction vector: ({dx}, {dy})")
            print(f"Step movement: ({move_dx}, {move_dy})")
            print(f"Current lead team orientation: {original_orientation}°")
            print(f"New orientation: {new_orientation}°")
            print(f"Rotation needed: {rotation_needed}°")

        # Track rotation states
        lead_team_rotated = False
        sl_rotated = False
        trail_team_rotated = False

        # Execute movement sequence
        for step in range(steps):
            if self.debug.enabled:
                self.debug.log_step_start(step, steps)

            # Move lead team
            old_lead_pos = lead_team.leader.soldier.position
            new_lead_pos = (
                old_lead_pos[0] + move_dx,
                old_lead_pos[1] + move_dy
            )

            # Check if lead team needs to rotate at start
            if not lead_team_rotated and abs(rotation_needed) > 1:
                if self.debug.enabled:
                    self.debug.log_team_rotation(lead_team, True)
                self.rotate_formation(lead_team, old_lead_pos, rotation_needed, "Lead Team")
                lead_team_rotated = True
                if self.debug.enabled:
                    self.debug.log_team_rotation(lead_team, False)

            # Move lead team as a unit
            delta_x = new_lead_pos[0] - old_lead_pos[0]
            delta_y = new_lead_pos[1] - old_lead_pos[1]

            if self.debug.enabled:
                self.debug.log_team_movement(lead_team, old_lead_pos, new_lead_pos)

            lead_team.leader.soldier.position = new_lead_pos
            for member in lead_team.members:
                member.soldier.position = (
                    member.soldier.position[0] + delta_x,
                    member.soldier.position[1] + delta_y
                )

            path_points.append(new_lead_pos)

            # Move SL
            if len(path_points) > 3:
                old_sl_pos = self.leader.position
                sl_target = path_points[-3]

                dist_to_pivot = self.manhattan_distance(old_sl_pos, pivot_point)
                if self.debug.enabled:
                    print(f"\nSL check (orientation: {self.orientation}°):")
                    print(f"  Distance to pivot: {dist_to_pivot}")
                    print(f"  Current position: {old_sl_pos}")
                    print(f"  Pivot point: {pivot_point}")

                if not sl_rotated and dist_to_pivot <= 3:
                    if self.debug.enabled:
                        print(f"  *** SL reached pivot point - executing rotation ***")
                    self.rotate_formation(self.leader, old_sl_pos, rotation_needed, "Squad Leader")
                    sl_rotated = True

                self.leader.position = sl_target
                if self.debug.enabled:
                    print(f"  Moved to: {sl_target}")

            # Move trail team
            if len(path_points) > 6:
                old_trail_pos = trail_team.leader.soldier.position
                trail_target = path_points[-6]

                dist_to_pivot = self.manhattan_distance(old_trail_pos, pivot_point)
                if self.debug.enabled:
                    self.debug.log_trail_team_status(trail_team, pivot_point, dist_to_pivot <= 3)

                if not trail_team_rotated and dist_to_pivot <= 3:
                    if self.debug.enabled:
                        self.debug.log_team_rotation(trail_team, True)
                    self.rotate_formation(trail_team, old_trail_pos, rotation_needed, "Trail Team")
                    trail_team_rotated = True
                    if self.debug.enabled:
                        self.debug.log_team_rotation(trail_team, False)

                # Move trail team as a unit
                delta_x = trail_target[0] - old_trail_pos[0]
                delta_y = trail_target[1] - old_trail_pos[1]

                trail_team.leader.soldier.position = trail_target
                for member in trail_team.members:
                    member.soldier.position = (
                        member.soldier.position[0] + delta_x,
                        member.soldier.position[1] + delta_y
                    )

            # Validate formation and capture frame
            if self.debug.enabled:
                self.debug.validate_formation(self, step + 1, steps, f"Movement Step {step + 1}")

            frames.append(self._capture_current_positions())

        # Final formation validation
        if self.debug.enabled:
            self.debug.log_formation_setup(self, "Final")
            movement_summary = self.debug.get_movement_summary()
            print("\nMovement Summary:", movement_summary)

        return frames

    def _execute_bounding(self, direction: Tuple[int, int], distance: int,
                          bound_distance: int = 50) -> List[Dict]:
        """
        Execute bounding movement technique, with initial squad rotation and proper bound execution.
        Includes consolidation of teams before changing direction.
        """
        frames = []
        current_distance = 0

        # Initialize debugging if enabled
        if self.debug.enabled:
            self.debug.clear_path_data()
            self.debug.log_movement_start(direction, distance, steps=distance // bound_distance + 1)
            self.debug.log_formation_setup(self, "Initial")
            print("\nStarting Bounding Movement:")
            print(f"Total distance: {distance}")
            print(f"Bound distance: {bound_distance}")

        frames.append(self._capture_current_positions())

        # Check if teams need consolidation (if bravo team is too far from alpha)
        alpha_pos = self.alpha_team.leader.soldier.position
        bravo_pos = self.bravo_team.leader.soldier.position
        consolidation_needed = self._distance(alpha_pos, bravo_pos) > bound_distance / 2

        if consolidation_needed:
            if self.debug.enabled:
                print("\nTeams spread out - consolidating before changing direction...")
                print(f"Alpha position: {alpha_pos}")
                print(f"Bravo position: {bravo_pos}")
                print(f"Current separation: {self._distance(alpha_pos, bravo_pos)}")

            # Calculate direction and distance for Bravo to rejoin Alpha
            dx = alpha_pos[0] - bravo_pos[0]
            dy = alpha_pos[1] - bravo_pos[1]
            rejoin_distance = int(math.sqrt(dx * dx + dy * dy))

            # Execute consolidation bound
            steps = 5  # Number of steps for smooth movement
            step_dx = dx // steps
            step_dy = dy // steps

            for step in range(steps):
                if self.debug.enabled:
                    self.debug.log_step_start(step, steps)
                    print(f"Consolidation step {step + 1}/{steps}")

                # Move Bravo team to Alpha's position
                old_bravo_pos = self.bravo_team.leader.soldier.position
                new_bravo_pos = (
                    old_bravo_pos[0] + step_dx,
                    old_bravo_pos[1] + step_dy
                )

                # Move the entire Bravo team
                delta_x = new_bravo_pos[0] - old_bravo_pos[0]
                delta_y = new_bravo_pos[1] - old_bravo_pos[1]

                if self.debug.enabled:
                    self.debug.log_team_movement(self.bravo_team, old_bravo_pos, new_bravo_pos)

                self.bravo_team.leader.soldier.position = new_bravo_pos
                for member in self.bravo_team.members:
                    member.soldier.position = (
                        member.soldier.position[0] + delta_x,
                        member.soldier.position[1] + delta_y
                    )

                frames.append(self._capture_current_positions())

            if self.debug.enabled:
                print("Teams consolidated. Ready for direction change.")

        # Calculate movement parameters
        dx, dy = direction
        if dx == 0 and dy == 0:
            return frames

        magnitude = int(math.sqrt(dx * dx + dy * dy))
        new_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)

        # Initial squad rotation to face movement direction
        rotation_needed = ((new_orientation - self.orientation + 180) % 360) - 180
        if abs(rotation_needed) > 1:
            if self.debug.enabled:
                print(f"\nInitial Squad Rotation:")
                print(f"Current orientation: {self.orientation}°")
                print(f"Target orientation: {new_orientation}°")
                print(f"Rotation needed: {rotation_needed}°")

            # Rotate squad leader
            pivot_pos = self.leader.position
            self.rotate_formation(self.leader, pivot_pos, rotation_needed, "Squad Leader")

            # Rotate both teams
            self.rotate_formation(self.alpha_team, self.alpha_team.leader.soldier.position,
                                  rotation_needed, "Alpha Team")
            self.rotate_formation(self.bravo_team, self.bravo_team.leader.soldier.position,
                                  rotation_needed, "Bravo Team")

            if self.debug.enabled:
                print(f"Squad rotation complete. New orientation: {self.orientation}°")
                self.debug.validate_formation(self, 0, 1, "Post-Rotation")

            # Adjust team positions based on formation template after rotation
            if self.debug.enabled:
                print("\nAdjusting team positions to maintain formation...")

            # Get squad leader's position as reference
            sl_pos = self.leader.position

            # Get template positions for teams
            alpha_template = self.formation_positions.get("Alpha Team", (0, 0))
            bravo_template = self.formation_positions.get("Bravo Team", (0, 0))

            # Calculate new positions based on rotated template
            angle_rad = math.radians(self.orientation - 90)  # Adjust to face movement direction

            # Calculate Alpha team's new position
            alpha_x = int(alpha_template[0] * math.cos(angle_rad) -
                          alpha_template[1] * math.sin(angle_rad))
            alpha_y = int(alpha_template[0] * math.sin(angle_rad) +
                          alpha_template[1] * math.cos(angle_rad))
            alpha_new_pos = (sl_pos[0] + alpha_x, sl_pos[1] + alpha_y)

            # Calculate Bravo team's new position
            bravo_x = int(bravo_template[0] * math.cos(angle_rad) -
                          bravo_template[1] * math.sin(angle_rad))
            bravo_y = int(bravo_template[0] * math.sin(angle_rad) +
                          bravo_template[1] * math.cos(angle_rad))
            bravo_new_pos = (sl_pos[0] + bravo_x, sl_pos[1] + bravo_y)

            # Move teams to adjusted positions
            if self.debug.enabled:
                print(f"Adjusting Alpha team position: {self.alpha_team.leader.soldier.position} -> {alpha_new_pos}")
                print(f"Adjusting Bravo team position: {self.bravo_team.leader.soldier.position} -> {bravo_new_pos}")

            # Move Alpha team
            delta_x = alpha_new_pos[0] - self.alpha_team.leader.soldier.position[0]
            delta_y = alpha_new_pos[1] - self.alpha_team.leader.soldier.position[1]
            self.alpha_team.leader.soldier.position = alpha_new_pos
            for member in self.alpha_team.members:
                member.soldier.position = (
                    member.soldier.position[0] + delta_x,
                    member.soldier.position[1] + delta_y
                )

            # Move Bravo team
            delta_x = bravo_new_pos[0] - self.bravo_team.leader.soldier.position[0]
            delta_y = bravo_new_pos[1] - self.bravo_team.leader.soldier.position[1]
            self.bravo_team.leader.soldier.position = bravo_new_pos
            for member in self.bravo_team.members:
                member.soldier.position = (
                    member.soldier.position[0] + delta_x,
                    member.soldier.position[1] + delta_y
                )

            if self.debug.enabled:
                self.debug.validate_formation(self, 0, 1, "Post-Position-Adjustment")

            frames.append(self._capture_current_positions())

        # Start with alpha team bounding
        bound_team = self.alpha_team
        overwatch_team = self.bravo_team
        bound_number = 1

        while current_distance < distance:
            # Store initial positions before each bound
            initial_bound_pos = bound_team.leader.soldier.position
            initial_overwatch_pos = overwatch_team.leader.soldier.position

            if self.debug.enabled:
                print(f"\n=== Bound {bound_number} ===")
                print(f"Distance covered: {current_distance}/{distance}")
                print(f"Bounding team: {bound_team.team_id}")
                print(f"Overwatch team: {overwatch_team.team_id}")
                print(f"\nInitial positions:")
                print(f"Bounding team ({bound_team.team_id}): {initial_bound_pos}")
                print(f"Overwatch team ({overwatch_team.team_id}): {initial_overwatch_pos}")

            # Calculate current bound distance
            remaining = distance - current_distance
            current_bound = min(bound_distance, remaining)

            # Execute bound movement in steps
            steps = 5  # Number of steps for smooth bound movement
            step_dx = (dx * current_bound) // (magnitude * steps)
            step_dy = (dy * current_bound) // (magnitude * steps)

            for step in range(steps):
                if self.debug.enabled:
                    self.debug.log_step_start(step, steps)
                    print(f"Moving bounding team step {step + 1}/{steps}")

                # Move bounding team
                old_bound_pos = bound_team.leader.soldier.position
                new_bound_pos = (
                    old_bound_pos[0] + step_dx,
                    old_bound_pos[1] + step_dy
                )

                # Move the entire bounding team as a unit
                delta_x = new_bound_pos[0] - old_bound_pos[0]
                delta_y = new_bound_pos[1] - old_bound_pos[1]

                if self.debug.enabled:
                    self.debug.log_team_movement(bound_team, old_bound_pos, new_bound_pos)

                bound_team.leader.soldier.position = new_bound_pos
                for member in bound_team.members:
                    member.soldier.position = (
                        member.soldier.position[0] + delta_x,
                        member.soldier.position[1] + delta_y
                    )

                # Update squad leader position if needed
                if bound_team == self.alpha_team:  # Squad leader moves with Alpha team
                    old_sl_x, old_sl_y = self.leader.position  # Unpack the tuple
                    new_sl_pos = (
                        old_sl_x + delta_x,
                        old_sl_y + delta_y
                    )
                    if self.debug.enabled:
                        print(f"Squad Leader moved with Alpha team: {self.leader.position} -> {new_sl_pos}")
                    self.leader.position = new_sl_pos

                # Validate formations and capture frame
                if self.debug.enabled:
                    bound_pos = bound_team.leader.soldier.position
                    overwatch_pos = overwatch_team.leader.soldier.position
                    print(f"\nCurrent positions:")
                    print(f"Bounding team ({bound_team.team_id}): {bound_pos}")
                    print(f"Overwatch team ({overwatch_team.team_id}): {overwatch_pos}")
                    self.debug.validate_formation(self, step + 1, steps,
                                                  f"Bound {bound_number} - Step {step + 1}")

                frames.append(self._capture_current_positions())

            # Update distance covered
            current_distance += current_bound

            if self.debug.enabled:
                final_bound_pos = bound_team.leader.soldier.position
                movement_distance = self._distance(initial_bound_pos, final_bound_pos)
                print(f"\nBound {bound_number} complete:")
                print(f"Distance moved: {movement_distance}")
                print(f"Total distance covered: {current_distance}/{distance}")

            # If we haven't reached the objective, swap roles
            if current_distance < distance:
                if self.debug.enabled:
                    print("\nSwapping bounding/overwatch roles")
                bound_team, overwatch_team = overwatch_team, bound_team
                bound_number += 1

        # Final formation validation
        if self.debug.enabled:
            self.debug.log_formation_setup(self, "Final")
            movement_summary = self.debug.get_movement_summary()
            print("\nBounding Movement Summary:", movement_summary)

        return frames

    def rotate_formation(self, unit: Union[Team, Soldier], pivot_pos: Tuple[int, int],
                         rotation_angle: int, formation_type: str) -> None:
        """
        Helper function for rotating squad elements around pivot points during follow-leader movement.

        Args:
            unit: The unit (Team or Soldier) to rotate
            pivot_pos: Position to rotate around
            rotation_angle: Angle of rotation in degrees
            formation_type: String identifying the type of formation (for logging)
        """
        if self.debug.enabled:
            print(f"\nRotating {formation_type} at {pivot_pos}")

        if isinstance(unit, Team):
            # Store original values for debug logging
            original_orientation = unit.orientation

            if self.debug.enabled:
                print(f"Before rotation: orientation = {original_orientation}°")
                print(f"Initial positions:")
                print(f"  TL: {unit.leader.soldier.position}")
                for member in unit.members:
                    print(f"  {member.soldier.role}: {member.soldier.position}")

            # Update team orientation first
            unit.orientation = (unit.orientation + rotation_angle) % 360
            angle_rad = math.radians(unit.orientation - 90)  # Adjust to face movement direction

            # Get team leader's position
            tl_pos = unit.leader.soldier.position

            # Rotate team members based on formation template
            for member in unit.members:
                # Get template position for this member
                template_pos = unit.formation_positions[member.soldier.role]

                # Calculate rotated position using template and team orientation
                rot_x = int(template_pos[0] * math.cos(angle_rad) -
                            template_pos[1] * math.sin(angle_rad))
                rot_y = int(template_pos[0] * math.sin(angle_rad) +
                            template_pos[1] * math.cos(angle_rad))

                # Apply rotated position relative to team leader
                member.soldier.position = (tl_pos[0] + rot_x, tl_pos[1] + rot_y)

                if self.debug.enabled:
                    print(f"  {member.soldier.role} rotated: {member.soldier.position}")

            if self.debug.enabled:
                print(f"After rotation: orientation = {unit.orientation}°")

                # Validate formation after rotation
                template_positions = {
                    member.soldier.role: unit.formation_positions[member.soldier.role]
                    for member in unit.members
                }
                current_positions = {
                    member.soldier.role: (
                        member.soldier.position[0] - tl_pos[0],
                        member.soldier.position[1] - tl_pos[1]
                    )
                    for member in unit.members
                }

                print("\nFormation validation after rotation:")
                for role in template_positions:
                    template = template_positions[role]
                    current = current_positions[role]
                    diff_x = abs(template[0] - current[0])
                    diff_y = abs(template[1] - current[1])
                    if diff_x > 1 or diff_y > 1:  # 1 unit tolerance
                        print(f"  WARNING: {role} position mismatch!")
                        print(f"    Template: {template}")
                        print(f"    Current: {current}")
                        print(f"    Difference: ({diff_x}, {diff_y})")

        else:  # Soldier (Squad Leader)
            if self.debug.enabled:
                print(f"Before rotation: squad orientation = {self.orientation}°")

            # For SL, we update the squad's orientation instead of moving the individual soldier
            self.orientation = (self.orientation + rotation_angle) % 360

            if self.debug.enabled:
                print(f"After rotation: squad orientation = {self.orientation}°")

    def manhattan_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> int:
        """Calculate Manhattan distance between two points."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos1[1])

    def _move_team_members(self, team: Team):
        """Move team members with their TL while maintaining formation orientation."""
        tl_pos = team.leader.soldier.position

        for member in team.members:
            # Get template position for this member
            template_pos = team.formation_positions[member.soldier.role]

            # Apply current team orientation to template position
            member_angle_rad = math.radians(team.orientation - 90)
            member_rot_x = int(template_pos[0] * math.cos(member_angle_rad) -
                               template_pos[1] * math.sin(member_angle_rad))
            member_rot_y = int(template_pos[0] * math.sin(member_angle_rad) +
                               template_pos[1] * math.cos(member_angle_rad))

            # Update member position relative to TL
            member.soldier.position = (
                tl_pos[0] + member_rot_x,
                tl_pos[1] + member_rot_y
            )

    def _distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """Calculate distance between two points."""
        return math.sqrt((pos2[0] - pos1[0]) ** 2 + (pos2[1] - pos1[1]) ** 2)

    def _capture_current_positions(self) -> Dict:
        """Capture current positions of all squad members for animation."""
        return {
            'unit_type': 'Squad',
            'squad_id': self.squad_id,
            'leader': {
                'role': 'Squad Leader',
                'position': self.leader.position
            },
            'teams': {
                'Alpha': self.alpha_team._capture_current_positions(),
                'Bravo': self.bravo_team._capture_current_positions()
            }
        }


@dataclass
class SpecialTeam:
    team_id: str
    members: List[TeamMember] = field(default_factory=list)
    orientation: int = 0
    formation: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_formation: str = "gun_team"  # Default formation name
    formation_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    parent_unit: Optional['Platoon'] = None

    @property
    def leader(self) -> TeamMember:
        """First member is always the team leader."""
        return self.members[0] if self.members else None

    @property
    def all_members(self):
        return self.members

    @property
    def alive_members(self):
        return [member for member in self.members if member.soldier.is_alive]

    def add_member(self, role: str, primary_weapon: Weapon, secondary_weapon: Optional[Weapon],
                   observation_range: int, engagement_range: int, position: Tuple[int, int]):
        soldier = Soldier(
            role=role,
            health=100,
            max_health=100,
            primary_weapon=primary_weapon,
            secondary_weapon=secondary_weapon,
            observation_range=observation_range,
            engagement_range=engagement_range,
            position=position,
            is_leader=(len(self.members) == 0)  # First member added is the leader
        )
        member = TeamMember(soldier, self.team_id)
        self.members.append(member)

    def set_formation(self, formation_positions: Dict[str, Tuple[int, int]],
                      formation_name: str):
        """Set formation positions and name."""
        self.formation_positions = formation_positions
        self.current_formation = formation_name
        if self.leader:
            self.apply_formation(self.leader.soldier.position)

    def apply_formation(self, base_position: Tuple[int, int] = (0, 0)):
        """Apply current formation from the given base position."""
        base_x, base_y = base_position
        for member in self.members:
            rel_x, rel_y = self.formation_positions.get(member.soldier.role, (0, 0))
            member.soldier.position = (base_x + rel_x, base_y + rel_y)

    def move(self, direction: Tuple[int, int], distance: int):
        """Base movement function using integer math."""
        dx, dy = direction
        magnitude = int(math.sqrt(dx * dx + dy * dy))
        if magnitude == 0:
            return

        # Calculate new orientation
        new_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        rotation_angle = new_orientation - self.orientation

        # Store leader's initial position
        leader_x, leader_y = self.leader.soldier.position

        # Move the leader
        new_leader_x = leader_x + (dx * distance) // magnitude
        new_leader_y = leader_y + (dy * distance) // magnitude
        self.leader.soldier.position = (new_leader_x, new_leader_y)

        # Rotate and translate other members
        for member in self.members[1:]:  # Skip leader as it's already moved
            rel_x = member.soldier.position[0] - leader_x
            rel_y = member.soldier.position[1] - leader_y

            cos_val = int(math.cos(math.radians(rotation_angle)) * 1000)
            sin_val = int(math.sin(math.radians(rotation_angle)) * 1000)

            rot_x = (rel_x * cos_val - rel_y * sin_val) // 1000
            rot_y = (rel_x * sin_val + rel_y * cos_val) // 1000

            member.soldier.position = (new_leader_x + rot_x, new_leader_y + rot_y)

        self.orientation = new_orientation

    def move_to_point(self, target_position: Tuple[int, int]):
        """Move team to a specific point while maintaining formation."""
        current_x, current_y = self.leader.soldier.position
        target_x, target_y = target_position

        dx = target_x - current_x
        dy = target_y - current_y

        distance = int(math.sqrt(dx * dx + dy * dy))

        if distance == 0:
            return

        direction = (dx, dy)
        self.move(direction, distance)

    def execute_movement(self, direction: Tuple[int, int], distance: int,
                         technique: MovementTechnique = MovementTechnique.TRAVELING) -> List[Dict]:
        """Execute movement using specified technique."""
        frames = []
        steps = 10

        step_distance = distance // steps
        for _ in range(steps):
            self.move(direction, step_distance)
            frames.append(self._capture_current_positions())

        return frames

    def _capture_current_positions(self) -> Dict:
        """Capture current positions of all team members for animation."""
        return {
            'Leader': self.leader.soldier.position,
            'Members': [member.soldier.position for member in self.members[1:]]
        }


@dataclass
class Platoon:
    platoon_id: str
    leader: Soldier
    squads: List[Squad]
    gun_teams: List[SpecialTeam]
    javelin_teams: List[SpecialTeam]
    orientation: int = 0
    formation: str = field(default="platoon_column")  # Formation name
    current_formation: str = "platoon_column"  # Default formation name
    formation_positions: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # Formation positions

    @property
    def all_members(self):
        """Return all members of the platoon including leader, squads, and special teams."""
        members = [self.leader]
        for squad in self.squads:
            members.extend(squad.all_members)
        for team in self.gun_teams + self.javelin_teams:
            members.extend(team.all_members)
        return members

    @property
    def alive_members(self):
        """Return all living members of the platoon."""
        return [member for member in self.all_members if member.is_alive]

    def check_and_replace_leader(self):
        """Replace platoon leader if killed with the highest ranking surviving squad leader."""
        if not self.leader.is_alive:
            for squad in self.squads:
                if squad.leader.is_alive:
                    self.leader = squad.leader
                    self.leader.is_leader = True
                    print(f"New platoon leader: {squad.squad_id} leader")
                    squad.check_and_replace_leader()  # Replace the squad leader
                    return
            print(f"No available leaders for {self.platoon_id}")

    def set_formation(self,
                      formation_positions: Dict[str, Tuple[int, int]],
                      squad_formation: Dict[str, Tuple[int, int]],
                      team_formation_alpha: Dict[str, Tuple[int, int]],
                      team_formation_bravo: Dict[str, Tuple[int, int]],
                      gun_team_formation: Dict[str, Tuple[int, int]],
                      javelin_team_formation: Dict[str, Tuple[int, int]],
                      formation_name: str):
        """Set formation positions and names for platoon and all subordinate units."""
        self.formation_positions = formation_positions
        self.current_formation = formation_name

        # Set formations for regular squads
        for squad in self.squads:
            squad.set_formation(squad_formation,
                                team_formation_alpha,
                                team_formation_bravo,
                                "squad_column_team_wedge")

        # Set formations for gun teams
        for team in self.gun_teams:
            team.set_formation(gun_team_formation, "gun_team")

        # Set formations for javelin teams
        for team in self.javelin_teams:
            team.set_formation(javelin_team_formation, "javelin_team")

    def apply_formation(self, base_position: Tuple[int, int] = (0, 0)):
        """Apply current formation from the given base position."""
        base_x, base_y = base_position

        # Position platoon leader
        pl_x, pl_y = self.formation_positions.get("Platoon Leader", (0, 0))
        self.leader.position = (base_x + pl_x, base_y + pl_y)

        # Position regular squads
        for i, squad in enumerate(self.squads):
            squad_key = f"{i + 1}st Squad" if i == 0 else f"{i + 1}nd Squad" if i == 1 else f"{i + 1}rd Squad"
            squad_x, squad_y = self.formation_positions.get(squad_key, (0, 0))
            squad.apply_formation((base_x + squad_x, base_y + squad_y))

        # Position gun teams
        for i, team in enumerate(self.gun_teams):
            team_key = f"Gun Team {i + 1}"
            team_x, team_y = self.formation_positions.get(team_key, (0, 0))
            team.apply_formation((base_x + team_x, base_y + team_y))

        # Position javelin teams
        for i, team in enumerate(self.javelin_teams):
            team_key = f"Javelin Team {i + 1}"
            team_x, team_y = self.formation_positions.get(team_key, (0, 0))
            team.apply_formation((base_x + team_x, base_y + team_y))

    def move(self, direction: Tuple[int, int], distance: int):
        """Base movement function using integer math."""
        dx, dy = direction
        magnitude = int(math.sqrt(dx * dx + dy * dy))
        if magnitude == 0:
            return

        # Calculate new orientation
        new_orientation = int((math.degrees(math.atan2(dy, dx)) + 360) % 360)
        rotation_angle = new_orientation - self.orientation

        # Store leader's initial position
        leader_x, leader_y = self.leader.position

        # Move the leader
        new_leader_x = leader_x + (dx * distance) // magnitude
        new_leader_y = leader_y + (dy * distance) // magnitude
        self.leader.position = (new_leader_x, new_leader_y)

        # Process each squad
        for squad in self.squads:
            # Calculate relative position to platoon leader
            squad_leader = squad.leader
            rel_x = squad_leader.position[0] - leader_x
            rel_y = squad_leader.position[1] - leader_y

            # Calculate sine and cosine values multiplied by relative positions
            cos_val = int(math.cos(math.radians(rotation_angle)) * 1000)
            sin_val = int(math.sin(math.radians(rotation_angle)) * 1000)

            # Rotate relative position (using integer math)
            rot_x = (rel_x * cos_val - rel_y * sin_val) // 1000
            rot_y = (rel_x * sin_val + rel_y * cos_val) // 1000

            # Update squad position and orientation
            squad_leader.position = (new_leader_x + rot_x, new_leader_y + rot_y)
            squad.orientation = new_orientation
            squad.apply_formation()

        # Process gun teams
        for team in self.gun_teams:
            # Calculate relative position to platoon leader
            team_leader = team.leader.soldier
            rel_x = team_leader.position[0] - leader_x
            rel_y = team_leader.position[1] - leader_y

            cos_val = int(math.cos(math.radians(rotation_angle)) * 1000)
            sin_val = int(math.sin(math.radians(rotation_angle)) * 1000)

            rot_x = (rel_x * cos_val - rel_y * sin_val) // 1000
            rot_y = (rel_x * sin_val + rel_y * cos_val) // 1000

            team_leader.position = (new_leader_x + rot_x, new_leader_y + rot_y)
            team.orientation = new_orientation
            team.apply_formation()

        # Process javelin teams
        for team in self.javelin_teams:
            # Calculate relative position to platoon leader
            team_leader = team.leader.soldier
            rel_x = team_leader.position[0] - leader_x
            rel_y = team_leader.position[1] - leader_y

            cos_val = int(math.cos(math.radians(rotation_angle)) * 1000)
            sin_val = int(math.sin(math.radians(rotation_angle)) * 1000)

            rot_x = (rel_x * cos_val - rel_y * sin_val) // 1000
            rot_y = (rel_x * sin_val + rel_y * cos_val) // 1000

            team_leader.position = (new_leader_x + rot_x, new_leader_y + rot_y)
            team.orientation = new_orientation
            team.apply_formation()

        # Update platoon orientation
        self.orientation = new_orientation

    def move_squad(self, squad_id: str, direction: Tuple[int, int], distance: int):
        """Direct movement of a specific squad."""
        squad = next((s for s in self.squads if s.squad_id == squad_id), None)
        if squad:
            squad.move(direction, distance)

    def move_gun_team(self, team_id: str, direction: Tuple[int, int], distance: int):
        """Direct movement of a specific gun team."""
        team = next((t for t in self.gun_teams if t.team_id == team_id), None)
        if team:
            team.move(direction, distance)

    def move_javelin_team(self, team_id: str, direction: Tuple[int, int], distance: int):
        """Direct movement of a specific javelin team."""
        team = next((t for t in self.javelin_teams if t.team_id == team_id), None)
        if team:
            team.move(direction, distance)

    def get_formation_type(self) -> FormationType:
        """Get the type of the current formation."""
        wagon_wheel_formations = [
            "platoon_line_squad_line"
        ]
        return (FormationType.WAGON_WHEEL
                if self.formation in wagon_wheel_formations
                else FormationType.FOLLOW_LEADER)

    def execute_movement(self, direction: Tuple[int, int], distance: int,
                         technique: MovementTechnique = MovementTechnique.TRAVELING) -> List[Dict]:
        """Execute movement using specified technique."""
        if technique == MovementTechnique.TRAVELING:
            formation_type = self.get_formation_type()
            if formation_type == FormationType.WAGON_WHEEL:
                return self._execute_traveling_wagon_wheel(direction, distance)
            else:
                return self._execute_traveling_follow_leader(direction, distance)
        else:  # BOUNDING
            return self._execute_bounding(direction, distance)

    def _execute_traveling_wagon_wheel(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute traveling movement for wagon wheel formations."""
        frames = []
        steps = 10  # Number of intermediate steps for smooth animation

        step_distance = distance // steps
        for _ in range(steps):
            self.move(direction, step_distance)
            frames.append(self._capture_current_positions())

        return frames

    def _execute_traveling_follow_leader(self, direction: Tuple[int, int], distance: int) -> List[Dict]:
        """Execute traveling movement for follow the leader formations."""
        frames = []
        pivot_points = []

        # Store pivot point if rotation is needed
        original_orientation = self.orientation
        new_orientation = int((math.degrees(math.atan2(direction[1], direction[0])) + 360) % 360)
        if abs(new_orientation - original_orientation) > 1:
            pivot_points.append(self.leader.position)

        # Move leader
        self.move(direction, distance)
        frames.append(self._capture_current_positions())

        # Move squads through pivot points
        for squad in self.squads:
            for pivot in pivot_points:
                squad.move_to_point(pivot)
                frames.append(self._capture_current_positions())
                squad.orientation = new_orientation
                frames.append(self._capture_current_positions())

            squad.move(direction, distance)
            frames.append(self._capture_current_positions())

        # Move gun teams through pivot points
        for team in self.gun_teams:
            for pivot in pivot_points:
                team.move_to_point(pivot)
                frames.append(self._capture_current_positions())
                team.orientation = new_orientation
                frames.append(self._capture_current_positions())

            team.move(direction, distance)
            frames.append(self._capture_current_positions())

        # Move javelin teams through pivot points
        for team in self.javelin_teams:
            for pivot in pivot_points:
                team.move_to_point(pivot)
                frames.append(self._capture_current_positions())
                team.orientation = new_orientation
                frames.append(self._capture_current_positions())

            team.move(direction, distance)
            frames.append(self._capture_current_positions())

        return frames

    def _execute_bounding(self, direction: Tuple[int, int], distance: int,
                          bound_distance: int = 100) -> List[Dict]:
        """Execute bounding movement technique."""
        frames = []
        current_distance = 0
        bound_squads = [self.squads[0]]  # First squad bounds first
        overwatch_squads = self.squads[1:]  # Other squads provide overwatch

        # Gun teams and javelin teams will move with the last bound
        weapon_teams = self.gun_teams + self.javelin_teams

        while current_distance < distance:
            # Calculate next bound distance
            remaining = distance - current_distance
            current_bound = min(bound_distance, remaining)

            # Move bounding squads
            for squad in bound_squads:
                squad.move(direction, current_bound)
                frames.append(self._capture_current_positions())

            # On last bound, move weapon teams
            if remaining <= bound_distance:
                for team in weapon_teams:
                    team.move(direction, current_bound)
                    frames.append(self._capture_current_positions())

            # Swap roles for squads
            if len(bound_squads) == 1:
                bound_squads = overwatch_squads
                overwatch_squads = [self.squads[0]]
            else:
                bound_squads = [self.squads[0]]
                overwatch_squads = self.squads[1:]

            current_distance += current_bound

        return frames

    def _capture_current_positions(self) -> Dict:
        """Capture current positions of all platoon members for animation."""
        return {
            'unit_type': 'Platoon',
            'platoon_id': self.platoon_id,
            'leader': {
                'role': 'Platoon Leader',
                'position': self.leader.position
            },
            'squads': [squad._capture_current_positions() for squad in self.squads],
            'gun_teams': [team._capture_current_positions() for team in self.gun_teams],
            'javelin_teams': [team._capture_current_positions() for team in self.javelin_teams]
        }


# Weapon definitions
M4 = Weapon("M4 Rifle", 50, 210, 1, "rifle_fire", False)
M249 = Weapon("M249 LMG", 80, 600, 6, "lmg_fire", True, 6)
M320 = GrenadeLauncher("M320 Grenade Launcher", 35, 12, 1, "grenade_fire", True, 5)
M240B = Weapon("M240B", 180, 1000, 9, "mmg_fire", True, 6)
Javelin = AreaDamageWeapon("Javelin", 200, 3, 1, "javelin_fire", True, 5)


# Formation definitions
def team_wedge_right() -> Dict[str, Tuple[int, int]]:
    return {
        "Team Leader": (0, 0),
        "Automatic Rifleman": (-2, -2),
        "Grenadier": (2, -2),
        "Rifleman": (4, -4)
    }


def team_wedge_left() -> Dict[str, Tuple[int, int]]:
    return {
        "Team Leader": (0, 0),
        "Automatic Rifleman": (2, -2),
        "Grenadier": (-2, -2),
        "Rifleman": (-4, -4)
    }


def team_line_right() -> Dict[str, Tuple[int, int]]:
    return {
        "Team Leader": (0, 0),
        "Automatic Rifleman": (-2, 0),
        "Grenadier": (2, 0),
        "Rifleman": (4, 0)
    }


def team_line_left() -> Dict[str, Tuple[int, int]]:
    return {
        "Team Leader": (0, 0),
        "Automatic Rifleman": (2, 0),
        "Grenadier": (-2, 0),
        "Rifleman": (-4, 0)
    }


def team_column() -> Dict[str, Tuple[int, int]]:
    return {
        "Team Leader": (0, 0),
        "Automatic Rifleman": (0, -2),
        "Grenadier": (0, -4),
        "Rifleman": (0, -6)
    }


def gun_team() -> Dict[str, Tuple[int, int]]:
    return {
        "Gunner": (0, 0),
        "Assistant Gunner": (2, 0)
    }


def javelin_team() -> Dict[str, Tuple[int, int]]:
    return {
        "Javelin Gunner": (0, 0),
        "Javelin Assistant Gunner": (2, 0)
    }


def squad_column_team_wedge() -> Dict[str, Tuple[int, int]]:
    return {
        "Squad Leader": (0, 0),
        "Alpha Team": (0, 8),
        "Bravo Team": (0, -3)
    }


def squad_column_team_column() -> Dict[str, Tuple[int, int]]:
    return {
        "Squad Leader": (0, 0),
        "Alpha Team": (0, 8),
        "Bravo Team": (0, -2)
    }


def squad_line_team_wedge() -> Dict[str, Tuple[int, int]]:
    return {
        "Squad Leader": (0, 0),
        "Alpha Team": (-4, 4),
        "Bravo Team": (4, 4)
    }


def platoon_column() -> Dict[str, Tuple[int, int]]:
    return {
        "Platoon Leader": (0, 0),
        "1st Squad": (0, 11),
        "Gun Team 1": (2, -2),
        "Javelin Team 1": (-4, -2),
        "2nd Squad": (0, -15),
        "Gun Team 2": (-4, -25),
        "Javelin Team 2": (2, -25),
        "3rd Squad": (0, -36)
    }


def platoon_line_squad_line() -> Dict[str, Tuple[int, int]]:
    return {
        "Platoon Leader": (0, 0),
        "1st Squad": (-19, 3),
        "2nd Squad": (0, 3),
        "3rd Squad": (19, 3),
        "Gun Team 1": (-15, -1),
        "Gun Team 2": (13, -1),
        "Javelin Team 1": (-9, -1),
        "Javelin Team 2": (7, -1)
    }


def create_infantry_squad(squad_id: str, start_position: Tuple[int, int]) -> Squad:
    x, y = start_position
    sl = Soldier("Squad Leader", 100, 100, M4, None, 48, 50, (x, y), True)

    alpha_leader_soldier = Soldier("Team Leader", 100, 100, M4, None, 48, 50, (x + 5, y), True)
    alpha_leader = TeamMember(alpha_leader_soldier, "Alpha")
    alpha = Team("Alpha", alpha_leader)
    alpha.add_member("Grenadier", M4, M320, 48, 50, (x + 10, y))
    alpha.add_member("Rifleman", M4, None, 48, 50, (x + 15, y))
    alpha.add_member("Automatic Rifleman", M249, None, 48, 360, (x + 20, y))

    bravo_leader_soldier = Soldier("Team Leader", 100, 100, M4, None, 48, 50, (x + 5, y + 5), True)
    bravo_leader = TeamMember(bravo_leader_soldier, "Bravo")
    bravo = Team("Bravo", bravo_leader)
    bravo.add_member("Grenadier", M4, M320, 48, 50, (x + 10, y + 5))
    bravo.add_member("Rifleman", M4, None, 48, 50, (x + 15, y + 5))
    bravo.add_member("Automatic Rifleman", M249, None, 48, 360, (x + 20, y + 5))

    return Squad(squad_id, sl, alpha, bravo)


def create_infantry_platoon(platoon_id: str, start_position: Tuple[int, int]) -> Platoon:
    x, y = start_position
    pl = Soldier("Platoon Leader", 100, 100, M4, None, 48, 50, (x, y), True)
    squads = [create_infantry_squad(f"{platoon_id}-Squad{i + 1}", (x + i * 10, y + i * 10)) for i in range(3)]

    gun_teams = []
    for i in range(2):
        team = SpecialTeam(f"Gun Team {i + 1}")
        team.add_member("Gunner", M240B, None, 70, 370, (x, y))
        team.add_member("Assistant Gunner", M4, None, 70, 50, (x, y))
        gun_teams.append(team)

    javelin_teams = []
    for i in range(2):
        team = SpecialTeam(f"Javelin Team {i + 1}")
        team.add_member("Javelin Gunner", M4, Javelin, 70, 100, (x, y))
        team.add_member("Javelin Assistant Gunner", M4, None, 70, 50, (x, y))
        javelin_teams.append(team)

    return Platoon(platoon_id, pl, squads, gun_teams, javelin_teams)


# Test code for formations
def test_all_formations():
    print("\nTesting All Formation Levels:")

    def test_team_formations():
        print("\n=== Testing Team Formations ===")

        # Create standalone team with debug enabled
        start_position = (100, 100)
        team_leader_soldier = Soldier("Team Leader", 100, 100, M4, None, 80, 50, start_position, True)
        team_leader = TeamMember(team_leader_soldier, "Test")
        test_team = Team("Test", team_leader)
        test_team.debug.enabled = True  # Enable debug validation

        # Add other team members
        test_team.add_member("Automatic Rifleman", M249, None, 70, 360, start_position)
        test_team.add_member("Grenadier", M4, M320, 70, 50, start_position)
        test_team.add_member("Rifleman", M4, None, 70, 50, start_position)

        # Test each team formation
        for formation_name, formation_func in [
            ("Team Wedge Right", team_wedge_right()),
            ("Team Wedge Left", team_wedge_left()),
            ("Team Line Right", team_line_right()),
            ("Team Line Left", team_line_left()),
            ("Team Column", team_column())
        ]:
            print(f"\nTesting {formation_name}:")
            test_team.set_formation(formation_func, formation_name)
            test_team.apply_formation(start_position)
            plot_team(test_team, formation_name)

    def test_special_teams():
        print("\n=== Testing Special Teams Formations ===")

        # Create standalone gun team
        start_position = (100, 100)
        gunner = Soldier("Gunner", 100, 100, M240B, None, 70, 370, start_position, True)
        gun_team_member = TeamMember(gunner, "GunTeam1")
        test_gun_team = SpecialTeam("GunTeam1", [gun_team_member])  # Renamed variable to avoid conflict
        test_gun_team.add_member("Assistant Gunner", M4, None, 70, 50, start_position)

        # Create standalone javelin team
        javelin_gunner = Soldier("Javelin Gunner", 100, 100, M4, Javelin, 70, 100, start_position, True)
        javelin_team_member = TeamMember(javelin_gunner, "JavelinTeam1")
        test_javelin_team = SpecialTeam("JavelinTeam1", [javelin_team_member])  # Renamed variable to avoid conflict
        test_javelin_team.add_member("Javelin Assistant Gunner", M4, None, 70, 50, start_position)

        # Test gun team formation
        print("\nGun Team Formation:")
        test_gun_team.set_formation(gun_team(), "gun_team")  # Using the formation function directly
        test_gun_team.apply_formation(start_position)
        plot_special_team(test_gun_team, "Gun Team Formation")
        print_positions(test_gun_team)

        # Test javelin team formation
        print("\nJavelin Team Formation:")
        test_javelin_team.set_formation(javelin_team(), "javelin_team")  # Using the formation function directly
        test_javelin_team.apply_formation(start_position)
        plot_special_team(test_javelin_team, "Javelin Team Formation")
        print_positions(test_javelin_team)

    def test_squad_formations():
        print("\n=== Testing Squad Formations ===")

        # Create standalone squad
        start_position = (100, 100)
        squad = create_infantry_squad("TestSquad", start_position)

        # Test squad formations
        formations_to_test = [
            {
                "name": "Squad Column Team Wedge",
                "formation": squad_column_team_wedge(),
                "team_alpha": team_wedge_right(),
                "team_bravo": team_wedge_left(),
                "formation_name": "squad_column_team_wedge"
            },
            {
                "name": "Squad Line Team Wedge",
                "formation": squad_line_team_wedge(),
                "team_alpha": team_wedge_left(),
                "team_bravo": team_wedge_right(),
                "formation_name": "squad_line_team_wedge"
            }
        ]

        for formation in formations_to_test:
            print(f"\n{formation['name']}:")

            # Set squad formation with all required parameters
            squad.set_formation(
                formation_positions=formation["formation"],
                team_formation_alpha=formation["team_alpha"],
                team_formation_bravo=formation["team_bravo"],
                formation_name=formation["formation_name"]
            )

            # Apply formations
            squad.apply_formation(start_position)

            # Plot and print positions
            plot_squad(squad, formation["name"])
            print_positions(squad)

    def test_platoon_formations():
        print("\n=== Testing Platoon Formations ===")

        # Create standalone platoon
        start_position = (100, 100)
        platoon = create_infantry_platoon("TestPlatoon", start_position)

        # Test platoon formations
        formations_to_test = [
            {
                "name": "Platoon Column",
                "formation": platoon_column(),
                "squad_formation": squad_column_team_wedge(),
                "team_alpha": team_wedge_right(),
                "team_bravo": team_wedge_left(),
                "gun_team": gun_team(),
                "javelin_team": javelin_team(),
                "formation_name": "platoon_column"
            },
            {
                "name": "Platoon Line Squad Line",
                "formation": platoon_line_squad_line(),
                "squad_formation": squad_line_team_wedge(),
                "team_alpha": team_wedge_left(),
                "team_bravo": team_wedge_right(),
                "gun_team": gun_team(),
                "javelin_team": javelin_team(),
                "formation_name": "platoon_line_squad_line"
            }
        ]

        for formation in formations_to_test:
            print(f"\n{formation['name']}:")

            # Set platoon formation with all required parameters
            platoon.set_formation(
                formation_positions=formation["formation"],
                squad_formation=formation["squad_formation"],
                team_formation_alpha=formation["team_alpha"],
                team_formation_bravo=formation["team_bravo"],
                gun_team_formation=formation["gun_team"],
                javelin_team_formation=formation["javelin_team"],
                formation_name=formation["formation_name"]
            )

            # Apply formations
            platoon.apply_formation(start_position)

            # Plot and print positions
            plot_platoon(platoon, formation["name"])
            print_positions(platoon)

    def plot_team(team, title):
        plt.figure(figsize=(10, 10))

        # Plot team leader
        x, y = team.leader.soldier.position
        plt.plot(x, y, 'ro', markersize=10, label='Team Leader')
        plt.text(x + 0.5, y + 0.5, 'TL', fontsize=8)

        # Plot other team members
        for member in team.members:
            x, y = member.soldier.position
            plt.plot(x, y, 'bo', markersize=8)
            label = 'AR' if member.soldier.role == 'Automatic Rifleman' else \
                'GRN' if member.soldier.role == 'Grenadier' else 'RFL'
            plt.text(x + 0.5, y + 0.5, label, fontsize=8)

        plt.grid(True)
        plt.title(title)
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()

        # Set equal scaling and reasonable view limits
        plt.axis('equal')
        all_positions = [team.leader.soldier.position] + \
                        [m.soldier.position for m in team.members]
        min_x = min(pos[0] for pos in all_positions) - 2
        max_x = max(pos[0] for pos in all_positions) + 2
        min_y = min(pos[1] for pos in all_positions) - 2
        max_y = max(pos[1] for pos in all_positions) + 2
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.show()

    def plot_special_team(team, title):
        plt.figure(figsize=(10, 10))

        # Plot all members
        for member in team.members:
            x, y = member.soldier.position
            if member is team.members[0]:  # First member
                plt.plot(x, y, 'ro', markersize=10)
                plt.text(x + 0.5, y + 0.5,
                         'MG' if member.soldier.role == 'Gunner' else 'JG',
                         fontsize=8)
            else:  # Second member
                plt.plot(x, y, 'bo', markersize=8)
                plt.text(x + 0.5, y + 0.5,
                         'AG' if member.soldier.role == 'Assistant Gunner' else 'JAG',
                         fontsize=8)

        plt.grid(True)
        plt.title(title)
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.axis('equal')

        # Set view limits
        all_positions = [m.soldier.position for m in team.members]
        min_x = min(pos[0] for pos in all_positions) - 2
        max_x = max(pos[0] for pos in all_positions) + 2
        min_y = min(pos[1] for pos in all_positions) - 2
        max_y = max(pos[1] for pos in all_positions) + 2
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.show()

    def plot_squad(squad, title):
        plt.figure(figsize=(12, 12))

        # Plot squad leader
        x, y = squad.leader.position
        plt.plot(x, y, 'o', color='red', markersize=12, label='Squad Leader')
        plt.text(x + 0.5, y + 0.5, 'SL', fontsize=10)

        # Plot teams with different colors
        team_colors = {'Alpha': 'b', 'Bravo': 'g'}  # Using matplotlib color codes

        for team in [squad.alpha_team, squad.bravo_team]:
            color = team_colors[team.team_id]

            # Plot team leader
            x, y = team.leader.soldier.position
            plt.plot(x, y, 'o', color=color, markersize=10, label=f'{team.team_id} Team')
            plt.text(x + 0.5, y + 0.5, 'TL', fontsize=8)

            # Plot team members
            for member in team.members:
                x, y = member.soldier.position
                plt.plot(x, y, 'o', color=color, markersize=8)
                label = 'AR' if member.soldier.role == 'Automatic Rifleman' else \
                    'GRN' if member.soldier.role == 'Grenadier' else 'RFL'
                plt.text(x + 0.5, y + 0.5, label, fontsize=8)

        plt.grid(True)
        plt.title(title)
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.axis('equal')

        # Set view limits
        all_positions = [squad.leader.position]
        for team in [squad.alpha_team, squad.bravo_team]:
            all_positions.extend(member.soldier.position for member in team.all_members)

        min_x = min(pos[0] for pos in all_positions) - 2
        max_x = max(pos[0] for pos in all_positions) + 2
        min_y = min(pos[1] for pos in all_positions) - 2
        max_y = max(pos[1] for pos in all_positions) + 2
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.show()

    def plot_platoon(platoon, title):
        plt.figure(figsize=(15, 15))

        # Plot platoon leader
        x, y = platoon.leader.position
        plt.plot(x, y, 'o', color='black', markersize=12, label='Platoon Leader')
        plt.text(x + 0.5, y + 0.5, 'PL', fontsize=10)

        # Plot squads with different colors
        squad_colors = ['r', 'b', 'g']  # Using matplotlib color codes
        for squad, color in zip(platoon.squads, squad_colors):
            # Plot squad leader
            x, y = squad.leader.position
            plt.plot(x, y, 'o', color=color, markersize=10, label=f'Squad {squad.squad_id}')
            plt.text(x + 0.5, y + 0.5, 'SL', fontsize=8)

            # Plot squad teams
            for team in [squad.alpha_team, squad.bravo_team]:
                # Plot team leader
                x, y = team.leader.soldier.position
                plt.plot(x, y, 'o', color=color, markersize=8)
                plt.text(x + 0.5, y + 0.5, 'TL', fontsize=8)

                # Plot team members
                for member in team.members:
                    x, y = member.soldier.position
                    plt.plot(x, y, 'o', color=color, markersize=8)
                    label = 'AR' if member.soldier.role == 'Automatic Rifleman' else \
                        'GRN' if member.soldier.role == 'Grenadier' else 'RFL'
                    plt.text(x + 0.5, y + 0.5, label, fontsize=8)

        # Plot gun teams
        for i, team in enumerate(platoon.gun_teams):
            for j, member in enumerate(team.members):
                x, y = member.soldier.position
                plt.plot(x, y, 'o', color='m', markersize=8,
                         label=f'Gun Team {i + 1}' if j == 0 else "")
                label = 'MG' if member.soldier.role == 'Gunner' else 'AG'
                plt.text(x + 0.5, y + 0.5, label, fontsize=8)

        # Plot javelin teams
        for i, team in enumerate(platoon.javelin_teams):
            for j, member in enumerate(team.members):
                x, y = member.soldier.position
                plt.plot(x, y, 'o', color='y', markersize=8,
                         label=f'Javelin Team {i + 1}' if j == 0 else "")
                label = 'JG' if member.soldier.role == 'Javelin Gunner' else 'JAG'
                plt.text(x + 0.5, y + 0.5, label, fontsize=8)

        plt.grid(True)
        plt.title(title)
        plt.xlabel("X coordinate")
        plt.ylabel("Y coordinate")
        plt.legend()
        plt.axis('equal')

        # Set view limits
        all_positions = [platoon.leader.position]
        for squad in platoon.squads:
            all_positions.append(squad.leader.position)
            for team in [squad.alpha_team, squad.bravo_team]:
                all_positions.extend(member.soldier.position for member in team.all_members)
        for team in platoon.gun_teams + platoon.javelin_teams:
            all_positions.extend(member.soldier.position for member in team.members)

        min_x = min(pos[0] for pos in all_positions) - 2
        max_x = max(pos[0] for pos in all_positions) + 2
        min_y = min(pos[1] for pos in all_positions) - 2
        max_y = max(pos[1] for pos in all_positions) + 2
        plt.xlim(min_x, max_x)
        plt.ylim(min_y, max_y)

        plt.show()

    def print_positions(unit):
        """Print positions for any unit type."""
        if isinstance(unit, Team):
            print(f"Team ID: {unit.team_id}")
            print(f"Team Leader: {unit.leader.soldier.position}")
            for member in unit.members:
                print(f"{member.soldier.role}: {member.soldier.position}")

        elif isinstance(unit, SpecialTeam):
            print(f"Special Team ID: {unit.team_id}")
            print(f"Team Leader: {unit.leader.soldier.position}")
            for member in unit.members:
                print(f"{member.soldier.role}: {member.soldier.position}")

        elif isinstance(unit, Squad):
            print(f"Squad ID: {unit.squad_id}")
            print(f"Squad Leader: {unit.leader.position}")
            print("\nAlpha Team:")
            print_positions(unit.alpha_team)
            print("\nBravo Team:")
            print_positions(unit.bravo_team)

        elif isinstance(unit, Platoon):
            print(f"Platoon ID: {unit.platoon_id}")
            print(f"Platoon Leader: {unit.leader.position}")

            for i, squad in enumerate(unit.squads):
                print(f"\nSquad {i + 1}:")
                print_positions(squad)

            print("\nGun Teams:")
            for i, team in enumerate(unit.gun_teams):
                print(f"Gun Team {i + 1}:")
                print_positions(team)

            print("\nJavelin Teams:")
            for i, team in enumerate(unit.javelin_teams):
                print(f"Javelin Team {i + 1}:")
                print_positions(team)

    # Run all tests in order
    test_team_formations()
    test_special_teams()
    test_squad_formations()
    test_platoon_formations()


# Test code for movement
def test_movements():
    """Test different movement types for each echelon level."""
    print("\nTesting All Movement Types:")

    def test_team_movements():
        print("\n=== Testing Team Movements ===")

        # Create test platoon and get a team to test
        platoon = create_infantry_platoon("1stPlatoon", (100, 100))
        test_team = platoon.squads[0].alpha_team
        test_team.orientation = 0  # Explicitly set to face north
        test_team.debug.enabled = True

        print(f"\nInitial team orientation before first test: {test_team.orientation}°")

        # Test 1: Team Wedge (Wagon Wheel) Movement
        print("\nExecuting Team Wedge (Wagon Wheel) Movement...")
        test_team.set_formation(
            formation_positions=team_wedge_right(),
            formation_name="team_wedge"
        )
        test_team.apply_formation((100, 100))
        print(f"Team orientation after formation set: {test_team.orientation}°")

        frames = test_team.execute_movement(
            direction=(1, 1),
            distance=50,
            technique=MovementTechnique.TRAVELING
        )
        create_movement_animation(
            frames=frames,
            title="Team Wedge (Wagon Wheel) Movement",
            save_path="team_wagon_wheel.gif",
            view_bounds=(90, 160, 90, 160)
        )

        # Test 2: Team Wedge (Wagon Wheel) Zig-Zag Movement
        print("\nExecuting Team Wedge (Wagon Wheel) Movement (Zig-Zag Pattern)...")
        test_team.orientation = 0  # Reset orientation before second test
        test_team.set_formation(
            formation_positions=team_wedge_left(),
            formation_name="team_wedge"
        )
        test_team.apply_formation((100, 100))
        print(f"Team orientation before zigzag: {test_team.orientation}°")

        # Define zig-zag movement sequence
        movement_sequence = [
            ((1, 1), 20),  # Forward-right diagonal
            ((1, -1), 20),  # Forward-left diagonal
            ((1, 1), 20),  # Forward-right diagonal
            ((1, -1), 20)  # Forward-left diagonal
        ]

        # Execute movement sequence
        all_frames = []
        for direction, distance in movement_sequence:
            print(f"\nStarting movement: direction={direction}, distance={distance}")
            print(f"Current orientation: {test_team.orientation}°")
            frames = test_team.execute_movement(
                direction=direction,
                distance=distance,
                technique=MovementTechnique.TRAVELING
            )
            all_frames.extend(frames)

        create_movement_animation(
            frames=all_frames,
            title="Team Wedge (Wagon Wheel) Movement - Zig Zag",
            save_path="team_wagon_wheel_zigzag.gif",
            view_bounds=(90, 160, 90, 160)
        )

        # Test 3: Team Column (Follow Leader) Movement
        print("\nExecuting Team Column (Follow Leader) Movement...")
        test_team.orientation = 0  # Reset orientation before third test
        test_team.set_formation(
            formation_positions=team_column(),
            formation_name="team_column"
        )
        test_team.apply_formation((100, 100))
        print(f"Team orientation before follow leader: {test_team.orientation}°")

        # Move forward
        frames = test_team.execute_movement(
            direction=(0, 1),  # Move straight up
            distance=30,
            technique=MovementTechnique.TRAVELING
        )

        # Get current position after move
        leader_pos = test_team.leader.soldier.position
        print(f"Position after forward move: {leader_pos}")

        # Turn and move right
        frames.extend(test_team.execute_movement(
            direction=(1, 0),  # Move right
            distance=30,
            technique=MovementTechnique.TRAVELING
        ))

        # Get position after turn
        leader_pos = test_team.leader.soldier.position
        print(f"Position after right turn: {leader_pos}")

        # Move southeast
        frames.extend(test_team.execute_movement(
            direction=(1, -1),  # Move southeast
            distance=30,
            technique=MovementTechnique.TRAVELING
        ))

        create_movement_animation(
            frames=frames,
            title="Team Column (Follow Leader) Movement",
            save_path="team_follow_leader.gif",
            view_bounds=(90, 200, 90, 200)  # Expanded view bounds
        )

        # Disable debug for cleanup
        test_team.debug.enabled = False

    def test_squad_movements():
        """Test different movement types for squad level operations."""
        print("\n=== Testing Squad Movements ===")

        # Create test platoon and get a squad to test
        platoon = create_infantry_platoon("1stPlatoon", (100, 100))
        test_squad = platoon.squads[0]
        test_squad.debug.enabled = True  # Enable debug for tests

        # Test 1: Squad Wagon Wheel Movement
        print("\nExecuting Squad Wagon Wheel Movement...")
        test_squad.set_formation(
            formation_positions=squad_line_team_wedge(),
            team_formation_alpha=team_wedge_right(),
            team_formation_bravo=team_wedge_left(),
            formation_name="squad_line_team_wedge"
        )
        test_squad.apply_formation((100, 100))
        frames = test_squad.execute_movement(
            direction=(1, 1),
            distance=50,
            technique=MovementTechnique.TRAVELING
        )
        create_movement_animation(
            frames=frames,
            title="Squad Wagon Wheel Movement",
            save_path="squad_wagon_wheel.gif",
            view_bounds=(90, 160, 90, 160)
        )

        # Test 2: Squad Follow Leader Movement
        print("\nExecuting Squad Follow Leader Movement...")
        test_squad.set_formation(
            formation_positions=squad_column_team_wedge(),
            team_formation_alpha=team_wedge_right(),
            team_formation_bravo=team_wedge_left(),
            formation_name="squad_column_team_wedge"
        )
        test_squad.apply_formation((100, 100))

        # First movement - North
        frames1 = test_squad.execute_movement(
            direction=(0, 1),
            distance=30,
            technique=MovementTechnique.TRAVELING
        )

        # Second movement - East
        frames2 = test_squad.execute_movement(
            direction=(1, 0),
            distance=30,
            technique=MovementTechnique.TRAVELING
        )

        # Third movement - Southeast
        frames3 = test_squad.execute_movement(
            direction=(1, 1),
            distance=30,
            technique=MovementTechnique.TRAVELING
        )

        # Combine all frames
        frames = frames1 + frames2 + frames3
        create_movement_animation(
            frames=frames,
            title="Squad Follow Leader Movement",
            save_path="squad_follow_leader.gif",
            view_bounds=(90, 160, 90, 160)
        )

        # Test 3: Squad Bounding Movement - North then Northeast
        print("\nExecuting Squad Bounding Movement Sequence...")
        test_squad.set_formation(
            formation_positions=squad_line_team_wedge(),
            team_formation_alpha=team_wedge_left(),
            team_formation_bravo=team_wedge_right(),
            formation_name="squad_line_team_wedge"
        )
        test_squad.apply_formation((100, 100))

        # First bound sequence - North
        print("\nExecuting bounds moving North...")
        frames_north = test_squad.execute_movement(
            direction=(0, 1),  # North
            distance=100,  # Total distance
            technique=MovementTechnique.BOUNDING
        )

        # Record frame of completion of north movement
        consolidation_frame = test_squad._capture_current_positions()
        frames_consolidate = [consolidation_frame]

        # Second bound sequence - Northeast
        print("\nExecuting bounds moving Northeast...")
        frames_northeast = test_squad.execute_movement(
            direction=(1, 1),  # Northeast
            distance=100,  # Total distance
            technique=MovementTechnique.BOUNDING
        )

        # Create individual movement animations
        create_movement_animation(
            frames=frames_north,
            title="Squad Bounding Movement - North",
            save_path="squad_bounding_north.gif",
            view_bounds=(90, 160, 90, 260)
        )

        create_movement_animation(
            frames=frames_northeast,
            title="Squad Bounding Movement - Northeast",
            save_path="squad_bounding_northeast.gif",
            view_bounds=(90, 260, 90, 260)
        )

        # Create combined movement animation with adjusted view bounds
        create_movement_animation(
            frames=frames_north + frames_consolidate + frames_northeast,
            title="Complete Squad Bounding Movement (North then Northeast)",
            save_path="squad_bounding_complete.gif",
            view_bounds=(90, 260, 90, 260)  # Adjusted to show full movement
        )

    def test_platoon_movements():
        print("\n=== Testing Platoon Movements ===")

        # Create test platoon
        platoon = create_infantry_platoon("1stPlatoon", (100, 100))

        # Test 1: Platoon Wagon Wheel Movement
        print("\nExecuting Platoon Wagon Wheel Movement...")
        platoon.set_formation(
            formation_positions=platoon_line_squad_line(),
            squad_formation=squad_line_team_wedge(),
            team_formation_alpha=team_wedge_right(),
            team_formation_bravo=team_wedge_left(),
            gun_team_formation=gun_team(),
            javelin_team_formation=javelin_team(),
            formation_name="platoon_line_squad_line"
        )
        platoon.apply_formation((100, 100))
        frames = platoon.execute_movement(
            direction=(1, 1),
            distance=50,
            technique=MovementTechnique.TRAVELING
        )
        create_movement_animation(
            frames=frames,
            title="Platoon Wagon Wheel Movement",
            save_path="platoon_wagon_wheel.gif",
            view_bounds=(90, 160, 90, 160)
        )

        # Test 2: Platoon Follow Leader Movement
        print("\nExecuting Platoon Follow Leader Movement...")
        platoon.set_formation(
            formation_positions=platoon_column(),
            squad_formation=squad_column_team_wedge(),
            team_formation_alpha=team_wedge_right(),
            team_formation_bravo=team_wedge_left(),
            gun_team_formation=gun_team(),
            javelin_team_formation=javelin_team(),
            formation_name="platoon_column"
        )
        platoon.apply_formation((100, 100))
        frames = platoon.execute_movement(
            direction=(1, 1),
            distance=50,
            technique=MovementTechnique.TRAVELING
        )
        create_movement_animation(
            frames=frames,
            title="Platoon Follow Leader Movement",
            save_path="platoon_follow_leader.gif",
            view_bounds=(90, 160, 90, 160)
        )

        # Test 3: Platoon Bounding Movement
        print("\nExecuting Platoon Bounding Movement...")
        platoon.set_formation(
            formation_positions=platoon_column(),
            squad_formation=squad_column_team_wedge(),
            team_formation_alpha=team_wedge_right(),
            team_formation_bravo=team_wedge_left(),
            gun_team_formation=gun_team(),
            javelin_team_formation=javelin_team(),
            formation_name="platoon_column"
        )
        platoon.apply_formation((100, 100))
        frames = platoon.execute_movement(
            direction=(1, 0),
            distance=200,
            technique=MovementTechnique.BOUNDING
        )
        create_movement_animation(
            frames=frames,
            title="Platoon Bounding Movement",
            save_path="platoon_bounding.gif",
            view_bounds=(90, 300, 90, 160)
        )

    # Run all test functions
    test_team_movements()
    test_squad_movements()
    test_platoon_movements()


def create_movement_animation(frames, title, save_path, view_bounds):
    """Create animation showing unit movements with role abbreviations."""
    print(f"\nCreating animation for: {title}")
    print(f"Number of frames: {len(frames)}")

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 15))

    # Role abbreviations mapping
    role_abbrev = {
        "Platoon Leader": "PL",
        "Squad Leader": "SL",
        "Team Leader": "TL",
        "Grenadier": "GRN",
        "Automatic Rifleman": "AR",
        "Rifleman": "RFLM",
        "Gunner": "MG",
        "Assistant Gunner": "AG",
        "Javelin Gunner": "JG",
        "Javelin Assistant Gunner": "JAG"
    }

    def init():
        ax.clear()
        return []

    def animate(frame):
        ax.clear()
        ax.grid(True)

        # Get unit type from frame
        unit_type = frame.get('unit_type')

        if unit_type == 'Platoon':
            # Plot platoon leader
            pl_pos = frame['leader']['position']
            ax.plot(pl_pos[0], pl_pos[1], 'ko', markersize=10)
            ax.text(pl_pos[0], pl_pos[1], 'PL', ha='center', va='center')

            # Plot squads
            for squad in frame['squads']:
                # Squad leader
                sl_pos = squad['leader']['position']
                ax.plot(sl_pos[0], sl_pos[1], 'ro', markersize=8)
                ax.text(sl_pos[0], sl_pos[1], 'SL', ha='center', va='center')

                # Plot teams
                for team_name, team in squad['teams'].items():
                    for member in team['positions']:
                        pos = member['position']
                        ax.plot(pos[0], pos[1], 'bo', markersize=6)
                        ax.text(pos[0], pos[1], role_abbrev[member['role']],
                                ha='center', va='center')

            # Plot special teams
            for team in frame['gun_teams'] + frame['javelin_teams']:
                for member in team['positions']:
                    pos = member['position']
                    ax.plot(pos[0], pos[1], 'go', markersize=6)
                    ax.text(pos[0], pos[1], role_abbrev[member['role']],
                            ha='center', va='center')

        elif unit_type == 'Squad':
            # Plot squad leader
            sl_pos = frame['leader']['position']
            ax.plot(sl_pos[0], sl_pos[1], 'ro', markersize=8)
            ax.text(sl_pos[0], sl_pos[1], 'SL', ha='center', va='center')

            # Plot teams
            for team_name, team in frame['teams'].items():
                for member in team['positions']:
                    pos = member['position']
                    ax.plot(pos[0], pos[1], 'bo', markersize=6)
                    ax.text(pos[0], pos[1], role_abbrev[member['role']],
                            ha='center', va='center')

        elif unit_type == 'Team':
            # Plot all team members
            for member in frame['positions']:
                pos = member['position']
                color = 'ro' if member['is_leader'] else 'bo'
                size = 8 if member['is_leader'] else 6
                ax.plot(pos[0], pos[1], color, markersize=size)
                ax.text(pos[0], pos[1], role_abbrev[member['role']],
                        ha='center', va='center')

        ax.set_xlim(view_bounds[0], view_bounds[1])
        ax.set_ylim(view_bounds[2], view_bounds[3])
        ax.set_aspect('equal')
        ax.set_title(f'{title}\nFrame {frames.index(frame) + 1}/{len(frames)}')

        return []

    try:
        # Create the animation
        print(f"Creating animation object...")
        anim = animation.FuncAnimation(fig, animate, init_func=init, frames=frames,
                                       interval=200, blit=True)

        # Save the animation
        print(f"Saving animation to: {save_path}")
        anim.save(save_path, writer='pillow')
        print(f"Animation saved successfully!")

        # Verify the file was created
        import os
        if os.path.exists(save_path):
            print(f"Verified: Animation file created at {save_path}")
            file_size = os.path.getsize(save_path)
            print(f"File size: {file_size / 1024:.2f} KB")
        else:
            print(f"Warning: Animation file not found at {save_path}")

    except Exception as e:
        print(f"Error creating/saving animation: {str(e)}")

    finally:
        # Clean up
        plt.close(fig)  # Close the figure to free memory
        print("Animation resources cleaned up")


# Example usage and testing
if __name__ == "__main__":
    # 1. Test platoon creation and initialization
    print("\n=== Testing Platoon Creation and Initialization ===")
    platoon = create_infantry_platoon("1stPlatoon", (100, 100))
    print(
        f"Platoon created with {len(platoon.squads)} squads, {len(platoon.gun_teams)} gun teams,"
        f" and {len(platoon.javelin_teams)} javelin teams")

    # Print detailed unit composition
    print("\nDetailed Unit Composition:")
    print(f"Platoon Leader: {platoon.leader.role}")
    for i, squad in enumerate(platoon.squads):
        print(f"\nSquad {i + 1}:")
        print(f"  Squad Leader: {squad.leader.role}")
        print("  Alpha Team:")
        for member in squad.alpha_team.all_members:
            print(f"    {member.soldier.role}")
        print("  Bravo Team:")
        for member in squad.bravo_team.all_members:
            print(f"    {member.soldier.role}")

    print("\nWeapons Squad:")
    for i, team in enumerate(platoon.gun_teams):
        print(f"  Gun Team {i + 1}:")
        for member in team.all_members:
            print(f"    {member.soldier.role}")
    for i, team in enumerate(platoon.javelin_teams):
        print(f"  Javelin Team {i + 1}:")
        for member in team.all_members:
            print(f"    {member.soldier.role}")

    # 2. Test formations
    print("\n=== Testing Formations ===")
    test_all_formations()

    # 3. Test succession of command
    print("\n=== Testing Succession of Command ===")

    # Test team leader succession
    print("\nTesting Team Leader Succession:")
    team = platoon.squads[0].alpha_team
    old_leader = team.leader
    print(f"Original team leader: {old_leader.soldier.role}")
    old_leader.soldier.health = 0
    team.check_and_replace_leader()
    print(f"New team leader: {team.leader.soldier.role}")

    # Test squad leader succession
    print("\nTesting Squad Leader Succession:")
    squad = platoon.squads[0]
    old_leader = squad.leader
    print(f"Original squad leader: {old_leader.role}")
    old_leader.health = 0
    squad.check_and_replace_leader()
    print(f"New squad leader: {squad.leader.role}")

    # Test platoon leader succession
    print("\nTesting Platoon Leader Succession:")
    old_leader = platoon.leader
    print(f"Original platoon leader: {old_leader.role}")
    old_leader.health = 0
    platoon.check_and_replace_leader()
    print(f"New platoon leader: {platoon.leader.role}")

    # 4. Test movements
    print("\n=== Testing Unit Movements ===")
    test_movements()

    print("\n=== Testing combat-related functionalities:")

    # Create some enemy soldiers for testing
    enemy1 = Soldier("Enemy1", 100, 100, Weapon("AK-47", 40, 180, 1, "enemy_rifle_fire", False), None, 50, 40,
                     (150, 150), False)
    enemy2 = Soldier("Enemy2", 100, 100, Weapon("AK-47", 40, 180, 1, "enemy_rifle_fire", False), None, 50, 40,
                     (160, 160), False)

    # Test firing weapons
    print("\nTesting weapon firing:")
    rifleman = next(member for member in platoon.all_members
                    if isinstance(member, TeamMember) and member.soldier.role == "Rifleman")
    initial_ammo = rifleman.soldier.primary_weapon.ammo_count
    targets = [enemy1, enemy2]

    for _ in range(5):  # Fire 5 times
        rifleman.soldier.fire_weapon(is_primary=True)

    print(
        f"Rifleman fired 5 times. Initial ammo: {initial_ammo},"
        f" Current ammo: {rifleman.soldier.primary_weapon.ammo_count}")
    assert rifleman.soldier.primary_weapon.ammo_count == initial_ammo - 5, "Ammo count not updated correctly"

    # Test damage calculation
    print("\nTesting damage calculation:")
    distance_to_enemy = int(math.sqrt((enemy1.position[0] - rifleman.soldier.position[0]) ** 2 +
                                      (enemy1.position[1] - rifleman.soldier.position[1]) ** 2))
    damage = rifleman.soldier.primary_weapon.calculate_damage(distance_to_enemy)
    print(f"Calculated damage at distance {distance_to_enemy}: {damage}")

    # Test hit probability
    print("\nTesting hit probability:")
    hit_prob = rifleman.soldier.primary_weapon.calculate_hit_probability(distance_to_enemy)
    print(f"Hit probability at distance {distance_to_enemy}: {hit_prob:.2f}")

    # Test applying damage to an enemy
    print("\nTesting damage application:")
    initial_enemy_health = enemy1.health
    enemy1.health -= damage
    print(f"Enemy health before: {initial_enemy_health}, after: {enemy1.health}")
    assert enemy1.health == initial_enemy_health - damage, "Damage not applied correctly"

    # Test firing a grenade launcher
    print("\nTesting grenade launcher:")
    grenadier = next(member for member in platoon.all_members
                     if isinstance(member, TeamMember) and member.soldier.role == "Grenadier")
    initial_grenade_ammo = grenadier.soldier.secondary_weapon.ammo_count
    grenadier.soldier.fire_weapon(is_primary=False)
    print(
        f"Grenadier fired grenade. Initial ammo: {initial_grenade_ammo},"
        f" Current ammo: {grenadier.soldier.secondary_weapon.ammo_count}")
    assert grenadier.soldier.secondary_weapon.ammo_count == initial_grenade_ammo - 1, \
        "Grenade ammo count not updated correctly"

    # Test area weapon (e.g., M249 LMG)
    print("\nTesting area weapon (M249 LMG):")
    auto_rifleman = next(member for member in platoon.all_members
                         if isinstance(member, TeamMember) and member.soldier.role == "Automatic Rifleman")
    initial_lmg_ammo = auto_rifleman.soldier.primary_weapon.ammo_count
    for _ in range(3):  # Fire 3 bursts
        auto_rifleman.soldier.fire_weapon(is_primary=True)
    print(
        f"Auto Rifleman fired 3 bursts. Initial ammo: {initial_lmg_ammo},"
        f" Current ammo: {auto_rifleman.soldier.primary_weapon.ammo_count}")
    assert auto_rifleman.soldier.primary_weapon.ammo_count == initial_lmg_ammo - (
            3 * auto_rifleman.soldier.primary_weapon.fire_rate), "LMG ammo count not updated correctly"

    print("\nAll combat-related tests completed successfully!")


""" 
To Do:
- Create Special Team movement functions.
- Create smoke.
- Standardize squad formations (alpha always first / right & bravo always second / left).
- Standardize debug outputs.
- Add descriptions to each function.
- Update file description.
"""
