"""
Tactical Route Analyzer Module - Updated Implementation

Purpose:
    Analyzes and determines optimal routes between start positions and tactical positions
    with enhanced tactical considerations and visualization capabilities.
"""

import os
import sys
import traceback
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set, Optional
from enum import Enum
import numpy as np
import math
from heapq import heappush, heappop
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Wedge
import logging

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

# Import from pathfinding module
from pathfinding import (
    TerrainInfo,
    TerrainType,
    ElevationType
)

# Import from tactical position analyzer
from tactical_position_analyzer import (
    TacticalPosition,
    TacticalPositionType,
    TacticalPositionAnalyzer,
    UnitSize
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


class RouteStrategy(Enum):
    """Defines different strategies for route planning."""
    SPEED = "speed"  # Fastest route
    COVER = "cover"  # Maximum cover
    CONCEALMENT = "concealment"  # Maximum concealment
    BALANCED = "balanced"  # Balanced consideration


class SquadElement(Enum):
    """Types of squad elements after split."""
    SUPPORT = "support"  # Support by fire element
    ASSAULT = "assault"  # Assault element
    FULL = "full"       # Full squad before split


@dataclass
class RouteSegment:
    """Represents a segment of a tactical route."""
    start_pos: Tuple[int, int]
    end_pos: Tuple[int, int]
    path: List[Tuple[int, int]]
    movement_technique: str  # 'traveling' or 'bounding'
    terrain_types: List[TerrainType]
    avg_cover: float
    avg_concealment: float
    threat_exposure: float

    @property
    def length(self) -> int:
        return len(self.path)


@dataclass
class ElementPositions:
    """Positions for different squad elements."""
    support_positions: List[TacticalPosition]  # Fire support/SBF positions
    assault_positions: List[TacticalPosition]  # Assault positions
    split_points: List[Tuple[int, int]]       # Potential squad split locations
    coordination_points: List[Tuple[int, int]] # Required coordination points


@dataclass
class CoordinationPoint:
    """Represents a coordination point for tactical movement."""
    position: Tuple[int, int]
    coordination_type: str  # 'support_set', 'phase_line', 'split', 'assault_start'
    required_elements: Set[SquadElement]  # Elements that must coordinate here
    conditions: List[str]  # Conditions that must be met
    priority: int  # Priority level (1 highest)

    def is_condition_met(self, element_positions: Dict[SquadElement, Tuple[int, int]]) -> bool:
        """Check if coordination conditions are met."""
        for condition in self.conditions:
            if condition == "support_set":
                if not self._check_support_set(element_positions):
                    return False
            elif condition == "mutual_support":
                if not self._check_mutual_support(element_positions):
                    return False
        return True

    def _check_support_set(self, element_positions: Dict[SquadElement, Tuple[int, int]]) -> bool:
        """Check if support element is in position."""
        return SquadElement.SUPPORT in element_positions

    def _check_mutual_support(self, element_positions: Dict[SquadElement, Tuple[int, int]]) -> bool:
        """Check if elements can provide mutual support."""
        # Implementation would check line of sight and support ranges
        pass


@dataclass
class MovementTiming:
    """Timing information for element movement."""
    start_time: float  # Minutes from H-hour
    end_time: float
    movement_rate: float  # Meters per minute
    distance: float  # Total distance in meters
    coordination_times: Dict[str, float]  # Time at each coordination point
    phase_durations: Dict[str, float]  # Duration of each movement phase


@dataclass
class CoordinationTiming:
    """Timing analysis for coordination points."""
    point: CoordinationPoint
    earliest_time: float
    latest_time: float
    required_duration: float
    dependencies: List[str]  # IDs of coordination points that must be complete first


@dataclass
class TacticalRoute:
    """Represents a complete tactical route with analysis."""
    segments: List[RouteSegment]
    total_distance: float
    avg_cover: float
    avg_concealment: float
    total_threat_exposure: float
    movement_time_estimate: float  # in minutes
    energy_expenditure: float  # relative measure
    quality_score: float

    @property
    def total_length(self) -> int:
        return sum(segment.length for segment in self.segments)


class TacticalRouteAnalyzer:
    """Main class for analyzing and planning tactical routes."""

    def __init__(self, terrain: List[List[TerrainInfo]], env_width: int, env_height: int):
        """Initialize the tactical route analyzer."""
        self.terrain = terrain
        self.env_width = env_width
        self.env_height = env_height
        self.threat_matrix = np.zeros((env_height, env_width))

        # Movement costs for different terrain (cells per minute)
        self.movement_speeds = {
            TerrainType.BARE: 5.0,        # 50m/min
            TerrainType.SPARSE_VEG: 4.0,  # 40m/min
            TerrainType.DENSE_VEG: 3.0,   # 30m/min
            TerrainType.WOODS: 2.0,       # 20m/min
            TerrainType.STRUCTURE: 1.0    # 10m/min
        }

        # Energy cost multipliers for terrain types
        self.energy_costs = {
            TerrainType.BARE: 1.0,
            TerrainType.SPARSE_VEG: 1.2,
            TerrainType.DENSE_VEG: 1.5,
            TerrainType.WOODS: 2.0,
            TerrainType.STRUCTURE: 2.5
        }

        self.position_analyzer = None
        self.element_positions = None

    def initialize_position_analyzer(self, objective: Tuple[int, int], terrain_analyzer=None):
        """Initialize position analyzer with objective."""
        self.position_analyzer = TacticalPositionAnalyzer(
            terrain_analyzer=terrain_analyzer,
            env_width=self.env_width,
            env_height=self.env_height,
            objective=objective
        )

    def analyze_squad_positions(self, objective: Tuple[int, int]) -> ElementPositions:
        """
        Analyze and identify positions for squad elements.

        Args:
            objective: Target objective position

        Returns:
            ElementPositions object containing positions for different elements
        """
        if not self.position_analyzer:
            self.initialize_position_analyzer(objective)

        # Find fire support positions (squad-sized initially)
        fire_support_positions = self.position_analyzer.find_tactical_positions(
            terrain=self.terrain,
            position_type=TacticalPositionType.FIRE_SUPPORT,
            unit_size=UnitSize.SQUAD,
            objective=objective,
            min_range=60,  # 600m
            max_range=150  # 1500m
        )

        # Find support by fire positions (team-sized)
        sbf_positions = self.position_analyzer.find_tactical_positions(
            terrain=self.terrain,
            position_type=TacticalPositionType.SUPPORT_BY_FIRE,
            unit_size=UnitSize.TEAM,
            objective=objective,
            min_range=40,  # 400m
            max_range=80  # 800m
        )

        # Find assault positions (team-sized)
        assault_positions = self.position_analyzer.find_tactical_positions(
            terrain=self.terrain,
            position_type=TacticalPositionType.ASSAULT,
            unit_size=UnitSize.TEAM,
            objective=objective,
            min_range=0,
            max_range=30  # 300m
        )

        # Identify potential split points
        split_points = self._identify_split_points(
            fire_support_positions + sbf_positions,
            assault_positions,
            objective
        )

        # Identify coordination points
        coordination_points = self._identify_coordination_points(
            split_points,
            fire_support_positions + sbf_positions,
            assault_positions
        )

        # Combine support positions and sort by quality
        support_positions = fire_support_positions + sbf_positions
        support_positions.sort(key=lambda x: x.quality_score, reverse=True)

        self.element_positions = ElementPositions(
            support_positions=support_positions,
            assault_positions=assault_positions,
            split_points=split_points,
            coordination_points=coordination_points
        )

        return self.element_positions

    def find_tactical_routes(
            self,
            start_pos: Tuple[int, int],
            objective: Tuple[int, int],
            strategy: RouteStrategy
    ) -> Dict[SquadElement, TacticalRoute]:
        """
        Find optimal routes for squad elements considering all tactical factors.

        Args:
            start_pos: Starting position
            objective: Final objective
            strategy: Route planning strategy

        Returns:
            Dict mapping squad elements to their routes
        """
        # First, analyze positions if not already done
        if not self.element_positions:
            self.analyze_squad_positions(objective)

        # Handle case where no split points were found
        if not self.element_positions.split_points:
            print("No split points found, calculating intermediate point...")
            # Calculate an intermediate point roughly halfway between start and first support position
            if self.element_positions.support_positions:
                support_pos = self.element_positions.support_positions[0].position
                split_point = self._calculate_intermediate_point(start_pos, support_pos)
                self.element_positions.split_points.append(split_point)
                print(f"Created intermediate split point at {split_point}")
            else:
                print("No support positions found, cannot calculate routes")
                return {}

        # Plan routes for each element
        try:
            # Full squad route to split point
            squad_route = self._plan_squad_route(
                start_pos,
                self.element_positions.split_points[0],
                strategy
            )

            # Support element route
            support_route = self._plan_support_route(
                self.element_positions.split_points[0],
                self.element_positions.support_positions,
                strategy
            )

            # Assault element route
            assault_route = self._plan_assault_route(
                self.element_positions.split_points[0],
                self.element_positions.assault_positions,
                objective,
                strategy
            )

            # Create TacticalRoute objects
            routes = {}

            if squad_route:
                routes[SquadElement.FULL] = self._create_tactical_route(
                    squad_route, SquadElement.FULL)

            if support_route:
                routes[SquadElement.SUPPORT] = self._create_tactical_route(
                    support_route, SquadElement.SUPPORT)

            if assault_route:
                routes[SquadElement.ASSAULT] = self._create_tactical_route(
                    assault_route, SquadElement.ASSAULT)

            return routes

        except Exception as e:
            print(f"Error planning routes: {str(e)}")
            raise

    def _create_tactical_route(self, path: List[Tuple[int, int]],
                              element: SquadElement) -> TacticalRoute:
        """
        Create TacticalRoute object from path.

        Args:
            path: List of positions
            element: Squad element type

        Returns:
            TacticalRoute object with analyzed characteristics
        """
        # Split path into segments
        segments = []
        segment_start = 0

        # Create segments at terrain/elevation changes
        for i in range(1, len(path)):
            prev_terrain = self.terrain[path[i-1][1]][path[i-1][0]].terrain_type
            curr_terrain = self.terrain[path[i][1]][path[i][0]].terrain_type

            if prev_terrain != curr_terrain:
                # Create segment
                segment = self._analyze_segment(
                    path[segment_start:i],
                    self._determine_movement_technique(
                        path[segment_start],
                        path[i-1],
                        element
                    )
                )
                segments.append(segment)
                segment_start = i

        # Add final segment
        if segment_start < len(path):
            segments.append(self._analyze_segment(
                path[segment_start:],
                self._determine_movement_technique(
                    path[segment_start],
                    path[-1],
                    element
                )
            ))

        # Calculate route characteristics
        total_distance = sum(len(seg.path) for seg in segments) * 10  # meters
        movement_time = self._estimate_movement_time(segments)
        energy = self._calculate_energy_expenditure(segments)

        # Calculate quality score
        quality_score = self._calculate_route_quality(path, element)

        return TacticalRoute(
            segments=segments,
            total_distance=total_distance,
            avg_cover=np.mean([seg.avg_cover for seg in segments]),
            avg_concealment=np.mean([seg.avg_concealment for seg in segments]),
            total_threat_exposure=np.mean([seg.threat_exposure for seg in segments]),
            movement_time_estimate=movement_time,
            energy_expenditure=energy,
            quality_score=quality_score
        )

    def _plan_squad_route(self,
                          start_pos: Tuple[int, int],
                          split_point: Tuple[int, int],
                          strategy: RouteStrategy) -> List[Tuple[int, int]]:
        """
        Plan route for full squad from start to split point.

        Args:
            start_pos: Starting position
            split_point: Squad split position
            strategy: Route planning strategy

        Returns:
            List of positions forming the route
        """
        # Use full squad movement considerations
        path = self._find_tactical_path(
            start=start_pos,
            end=split_point,
            strategy=strategy,
            element_type=SquadElement.FULL
        )

        if not path:
            print(f"Warning: No valid path found from {start_pos} to split point {split_point}")
            return []

        return path

    def _plan_support_route(self,
                            split_point: Tuple[int, int],
                            support_positions: List[TacticalPosition],
                            strategy: RouteStrategy) -> List[Tuple[int, int]]:
        """
        Plan route for support element from split point to support position.

        Args:
            split_point: Starting position after split
            support_positions: Potential support positions
            strategy: Route planning strategy

        Returns:
            List of positions forming the route
        """
        best_route = []
        best_score = -1

        for pos in support_positions:
            # Find path to this support position
            path = self._find_tactical_path(
                start=split_point,
                end=pos.position,
                strategy=strategy,
                element_type=SquadElement.SUPPORT
            )

            if path:
                # Score route
                score = self._calculate_route_quality(path, SquadElement.SUPPORT)
                if score > best_score:
                    best_score = score
                    best_route = path

        if not best_route:
            print("Warning: No valid support route found")

        return best_route

    def _plan_assault_route(self,
                            split_point: Tuple[int, int],
                            assault_positions: List[TacticalPosition],
                            objective: Tuple[int, int],
                            strategy: RouteStrategy) -> List[Tuple[int, int]]:
        """
        Plan route for assault element from split point through assault position to objective.

        Args:
            split_point: Starting position after split
            assault_positions: Potential assault positions
            objective: Final objective
            strategy: Route planning strategy

        Returns:
            List of positions forming complete assault route
        """
        best_route = []
        best_score = -1

        for pos in assault_positions:
            # Find path to assault position
            path1 = self._find_tactical_path(
                start=split_point,
                end=pos.position,
                strategy=strategy,
                element_type=SquadElement.ASSAULT
            )

            if path1:
                # Find path from assault position to objective
                path2 = self._find_tactical_path(
                    start=pos.position,
                    end=objective,
                    strategy=strategy,
                    element_type=SquadElement.ASSAULT
                )

                if path2:
                    # Combine paths and score route
                    full_path = path1 + path2[1:]  # Avoid duplicate position
                    score = self._calculate_route_quality(full_path, SquadElement.ASSAULT)
                    if score > best_score:
                        best_score = score
                        best_route = full_path

        if not best_route:
            print("Warning: No valid assault route found")

        return best_route

    def _identify_split_points(self,
                               support_positions: List[TacticalPosition],
                               assault_positions: List[TacticalPosition],
                               objective: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Identify potential squad split points based on tactical positions.

        Args:
            support_positions: List of potential support positions
            assault_positions: List of potential assault positions
            objective: Final objective position

        Returns:
            List of potential split point positions
        """
        split_points = []

        # Get average position of support positions
        if support_positions:
            support_x = sum(p.position[0] for p in support_positions) / len(support_positions)
            support_y = sum(p.position[1] for p in support_positions) / len(support_positions)
            support_center = (int(support_x), int(support_y))
        else:
            return []  # No split points without support positions

        # Get average position of assault positions
        if assault_positions:
            assault_x = sum(p.position[0] for p in assault_positions) / len(assault_positions)
            assault_y = sum(p.position[1] for p in assault_positions) / len(assault_positions)
            assault_center = (int(assault_x), int(assault_y))
        else:
            assault_center = objective

        # Calculate primary split point
        primary_x = (support_center[0] + assault_center[0]) // 2
        primary_y = (support_center[1] + assault_center[1]) // 2
        primary_split = (primary_x, primary_y)

        # Check terrain at primary split point
        if self._is_valid_split_point(primary_split):
            split_points.append(primary_split)

        # Add additional split points if needed
        offset = 10  # cells
        for dx, dy in [(offset, 0), (-offset, 0), (0, offset), (0, -offset)]:
            alt_point = (primary_x + dx, primary_y + dy)
            if self._is_valid_split_point(alt_point):
                split_points.append(alt_point)

        return split_points

    def _is_valid_split_point(self, position: Tuple[int, int]) -> bool:
        """Check if position is suitable for squad split."""
        x, y = position

        # Check bounds
        if not (0 <= x < self.env_width and 0 <= y < self.env_height):
            return False

        # Get terrain info
        terrain_info = self.terrain[y][x]

        # Check terrain suitability
        if terrain_info.terrain_type in [TerrainType.STRUCTURE, TerrainType.WOODS]:
            return False  # Too restrictive for squad split

        # Check threat level
        if self.threat_matrix[y][x] > 0.7:
            return False  # Too dangerous for split

        # Check nearby cover
        if not self._has_nearby_cover(position):
            return False  # Need some cover near split point

        return True

    def _has_nearby_cover(self, position: Tuple[int, int], radius: int = 3) -> bool:
        """Check if there is adequate cover near position."""
        x, y = position

        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                check_x, check_y = x + dx, y + dy
                if 0 <= check_x < self.env_width and 0 <= check_y < self.env_height:
                    terrain_info = self.terrain[check_y][check_x]
                    if terrain_info.cover_bonus >= 0.5:
                        return True

        return False

    def _calculate_intermediate_point(self,
                                      start: Tuple[int, int],
                                      end: Tuple[int, int]) -> Tuple[int, int]:
        """Calculate intermediate point between start and end positions."""
        mid_x = (start[0] + end[0]) // 2
        mid_y = (start[1] + end[1]) // 2

        # Adjust if point is not valid
        if not self._is_valid_split_point((mid_x, mid_y)):
            # Try points at different distances along the line
            for ratio in [0.4, 0.6, 0.3, 0.7]:
                x = int(start[0] + (end[0] - start[0]) * ratio)
                y = int(start[1] + (end[1] - start[1]) * ratio)
                if self._is_valid_split_point((x, y)):
                    return (x, y)

        return (mid_x, mid_y)

    def analyze_movement_timing(self,
                                routes: Dict[SquadElement, List[Tuple[int, int]]],
                                coordination_points: List[CoordinationPoint]) -> Dict[SquadElement, MovementTiming]:
        """
        Analyze timing for element movements and coordination.

        Args:
            routes: Dict mapping elements to their routes
            coordination_points: List of coordination points

        Returns:
            Dict mapping elements to their movement timing
        """
        # Calculate base movement timing
        movement_timing = {}

        for element, route in routes.items():
            # Calculate movement phases
            phases = self._identify_movement_phases(route, coordination_points)

            # Calculate timing for each phase
            phase_durations = {}
            coordination_times = {}
            total_time = 0.0

            for phase_name, phase_points in phases.items():
                duration = self._calculate_phase_duration(
                    phase_points,
                    element,
                    phase_name
                )
                phase_durations[phase_name] = duration
                total_time += duration

                # Record coordination point times
                for cp in coordination_points:
                    if any(self._is_at_position(point, cp.position, 5)
                           for point in phase_points):
                        coordination_times[cp.coordination_type] = total_time

            # Calculate overall movement metrics
            distance = self._calculate_route_distance(route)
            avg_rate = distance / total_time if total_time > 0 else 0

            movement_timing[element] = MovementTiming(
                start_time=0.0,  # Will adjust based on coordination
                end_time=total_time,
                movement_rate=avg_rate,
                distance=distance,
                coordination_times=coordination_times,
                phase_durations=phase_durations
            )

        # Adjust timings based on coordination requirements
        self._adjust_movement_timings(movement_timing, coordination_points)

        return movement_timing

    def _identify_movement_phases(self,
                                  route: List[Tuple[int, int]],
                                  coordination_points: List[CoordinationPoint]) -> Dict[str, List[Tuple[int, int]]]:
        """Break route into movement phases based on coordination points."""
        phases = {}
        current_phase = []
        current_phase_name = "initial_movement"

        for point in route:
            current_phase.append(point)

            # Check if point is at a coordination point
            for cp in coordination_points:
                if self._is_at_position(point, cp.position, 5):
                    # End current phase
                    if current_phase:
                        phases[current_phase_name] = current_phase

                    # Start new phase
                    current_phase = [point]
                    current_phase_name = f"post_{cp.coordination_type}"
                    break

        # Add final phase
        if current_phase:
            phases[current_phase_name] = current_phase

        return phases

    def _calculate_phase_duration(self,
                                  phase_points: List[Tuple[int, int]],
                                  element: SquadElement,
                                  phase_name: str) -> float:
        """Calculate duration of movement phase in minutes."""
        if not phase_points:
            return 0.0

        total_time = 0.0

        # Calculate base movement time
        for i in range(len(phase_points) - 1):
            current = phase_points[i]
            next_point = phase_points[i + 1]

            # Get terrain and threat factors
            terrain_factor = self._get_terrain_movement_factor(current)
            threat_factor = self._get_threat_movement_factor(current)

            # Calculate distance and time
            distance = self._calculate_distance(current, next_point)

            # Base movement rate (meters/minute) varies by element and phase
            base_rate = self._get_base_movement_rate(element, phase_name)

            # Apply factors
            adjusted_rate = base_rate * terrain_factor * threat_factor
            segment_time = distance / adjusted_rate if adjusted_rate > 0 else 0

            total_time += segment_time

        # Add coordination time if at coordination point
        if "post_" in phase_name:
            total_time += self._get_coordination_time(phase_name)

        return total_time

    def _get_base_movement_rate(self,
                                element: SquadElement,
                                phase_name: str) -> float:
        """Get base movement rate based on element and phase."""
        # Base rates in meters/minute
        rates = {
            SquadElement.FULL: {
                "initial_movement": 20.0,  # 200m/min
                "post_split": 15.0
            },
            SquadElement.SUPPORT: {
                "initial_movement": 15.0,
                "post_support_set": 10.0
            },
            SquadElement.ASSAULT: {
                "initial_movement": 15.0,
                "post_phase_line": 12.0,
                "post_assault_start": 8.0
            }
        }

        element_rates = rates.get(element, {})
        return element_rates.get(phase_name, 15.0)  # Default 15 m/min

    def _get_terrain_movement_factor(self, position: Tuple[int, int]) -> float:
        """Get movement factor based on terrain."""
        x, y = position
        terrain_info = self.terrain[y][x]

        # Factor based on terrain type
        terrain_factors = {
            TerrainType.BARE: 1.0,
            TerrainType.SPARSE_VEG: 0.9,
            TerrainType.DENSE_VEG: 0.7,
            TerrainType.WOODS: 0.5,
            TerrainType.STRUCTURE: 0.3
        }

        # Factor based on elevation
        elevation_factors = {
            ElevationType.GROUND_LEVEL: 1.0,
            ElevationType.ELEVATED_LEVEL: 0.7,
            ElevationType.LOWER_LEVEL: 0.8
        }

        return (terrain_factors.get(terrain_info.terrain_type, 1.0) *
                elevation_factors.get(terrain_info.elevation_type, 1.0))

    def _get_threat_movement_factor(self, position: Tuple[int, int]) -> float:
        """Get movement factor based on threat."""
        x, y = position
        threat_level = self.threat_matrix[y][x]

        # Higher threat = slower movement
        return max(0.2, 1.0 - (threat_level * 0.8))

    def _get_coordination_time(self, phase_name: str) -> float:
        """Get time required for coordination at phase transitions."""
        # Time in minutes for different coordination types
        coordination_times = {
            "post_support_set": 5.0,  # Time to set support position
            "post_split": 3.0,  # Time to coordinate split
            "post_phase_line": 2.0,  # Time at phase line
            "post_assault_start": 4.0  # Time to prep for assault
        }

        return coordination_times.get(phase_name, 0.0)

    def _adjust_movement_timings(self,
                                 movement_timing: Dict[SquadElement, MovementTiming],
                                 coordination_points: List[CoordinationPoint]):
        """Adjust movement timings to ensure proper coordination."""
        # Analyze coordination dependencies
        coordination_timing = self._analyze_coordination_timing(
            movement_timing, coordination_points)

        # Adjust start times to meet dependencies
        self._adjust_start_times(movement_timing, coordination_timing)

        # Update end times and coordination times
        for element, timing in movement_timing.items():
            timing.end_time = timing.start_time + sum(timing.phase_durations.values())

            # Adjust coordination times
            new_coord_times = {}
            current_time = timing.start_time
            for phase, duration in timing.phase_durations.items():
                if phase.startswith("post_"):
                    coord_type = phase.replace("post_", "")
                    new_coord_times[coord_type] = current_time + duration
                current_time += duration
            timing.coordination_times = new_coord_times

    def _analyze_coordination_timing(self,
                                     movement_timing: Dict[SquadElement, MovementTiming],
                                     coordination_points: List[CoordinationPoint]) -> List[CoordinationTiming]:
        """
        Analyze timing requirements for coordination points.

        Args:
            movement_timing: Movement timing for each element
            coordination_points: List of coordination points

        Returns:
            List of CoordinationTiming objects detailing requirements
        """
        coordination_timing = []

        for cp in coordination_points:
            # Find all elements that need to coordinate
            element_times = []
            for element, timing in movement_timing.items():
                if element in cp.required_elements:
                    coord_time = timing.coordination_times.get(cp.coordination_type)
                    if coord_time is not None:
                        element_times.append(coord_time)

            if element_times:
                # Calculate timing window
                earliest = min(element_times)
                latest = max(element_times)
                duration = self._get_coordination_time(f"post_{cp.coordination_type}")

                # Determine dependencies
                dependencies = []
                if cp.coordination_type == "assault_start":
                    dependencies.extend(["support_set", "phase_line"])
                elif cp.coordination_type == "phase_line":
                    dependencies.append("support_set")

                coordination_timing.append(CoordinationTiming(
                    point=cp,
                    earliest_time=earliest,
                    latest_time=latest,
                    required_duration=duration,
                    dependencies=dependencies
                ))

        return coordination_timing

    def _adjust_start_times(self,
                            movement_timing: Dict[SquadElement, MovementTiming],
                            coordination_timing: List[CoordinationTiming]):
        """
        Adjust element start times to meet coordination requirements.

        Args:
            movement_timing: Movement timing for each element
            coordination_timing: Coordination timing requirements
        """
        # Sort coordination points by dependencies
        coord_points = self._sort_coordination_points(coordination_timing)

        # Track latest required start time for each element
        required_starts = {element: 0.0 for element in movement_timing.keys()}

        # Process coordination points in order
        for ct in coord_points:
            # Find elements that need to coordinate
            coordinating_elements = ct.point.required_elements

            # Calculate required start times to meet coordination
            for element in coordinating_elements:
                timing = movement_timing[element]
                coord_time = timing.coordination_times.get(ct.point.coordination_type)
                if coord_time is not None:
                    # Calculate when element needs to start to meet coordination
                    required_start = ct.earliest_time - coord_time
                    required_starts[element] = max(
                        required_starts[element],
                        required_start
                    )

        # Apply adjusted start times
        for element, start_time in required_starts.items():
            movement_timing[element].start_time = start_time

    def _sort_coordination_points(self, coord_timing: List[CoordinationTiming]) -> List[CoordinationTiming]:
        """
        Sort coordination points based on dependencies and timing.

        Args:
            coord_timing: List of coordination timing objects

        Returns:
            Sorted list of coordination points
        """
        # Create dependency graph
        dependency_graph = defaultdict(list)
        for ct in coord_timing:
            for dep in ct.dependencies:
                dependency_graph[dep].append(ct.point.coordination_type)

        # Track processed points
        processed = set()
        sorted_points = []

        def process_point(point_type: str):
            if point_type in processed:
                return
            # Process dependencies first
            for dep in dependency_graph[point_type]:
                if dep not in processed:
                    process_point(dep)
            processed.add(point_type)
            # Add corresponding coordination timing
            ct = next((ct for ct in coord_timing
                       if ct.point.coordination_type == point_type), None)
            if ct:
                sorted_points.append(ct)

        # Process all points
        for ct in coord_timing:
            process_point(ct.point.coordination_type)

        return sorted_points

    def _calculate_route_quality(self, route: List[Tuple[int, int]], element: SquadElement) -> float:
        """
        Calculate quality score for a route based on element type and tactical considerations.

        Args:
            route: List of positions forming the route
            element: Type of squad element using the route

        Returns:
            Quality score between 0 and 1
        """
        if not route:
            return 0.0

        # Calculate route characteristics
        total_cover = 0.0
        total_concealment = 0.0
        total_threat = 0.0
        total_movement_cost = 0.0
        elevation_changes = 0
        prev_elevation = None

        for x, y in route:
            # Get terrain and threat info
            terrain_info = self.terrain[y][x]

            # Accumulate metrics
            total_cover += terrain_info.cover_bonus
            total_concealment += (1 - terrain_info.visibility_factor)
            total_movement_cost += terrain_info.movement_cost
            total_threat += self.threat_matrix[y][x]

            # Count elevation changes
            curr_elevation = terrain_info.elevation_type
            if prev_elevation and curr_elevation != prev_elevation:
                elevation_changes += 1
            prev_elevation = curr_elevation

        # Calculate averages
        route_length = len(route)
        avg_cover = total_cover / route_length
        avg_concealment = total_concealment / route_length
        avg_movement_cost = total_movement_cost / route_length
        avg_threat = total_threat / route_length

        # Normalize elevation changes
        elevation_factor = 1.0 - (min(elevation_changes, 5) / 5.0)

        # Different weighting based on element type
        if element == SquadElement.FULL:
            # Full squad prioritizes protection and steady movement
            return (avg_cover * 0.25 +
                    avg_concealment * 0.25 +
                    (1 - avg_movement_cost / 3.0) * 0.2 +
                    (1 - avg_threat) * 0.2 +
                    elevation_factor * 0.1)

        elif element == SquadElement.SUPPORT:
            # Support element prioritizes cover and stable positions
            return (avg_cover * 0.35 +
                    avg_concealment * 0.2 +
                    (1 - avg_movement_cost / 3.0) * 0.15 +
                    (1 - avg_threat) * 0.2 +
                    elevation_factor * 0.1)

        else:  # ASSAULT
            # Assault element prioritizes concealment and movement
            return (avg_cover * 0.2 +
                    avg_concealment * 0.35 +
                    (1 - avg_movement_cost / 3.0) * 0.25 +
                    (1 - avg_threat) * 0.15 +
                    elevation_factor * 0.05)

    def _calculate_route_distance(self, route: List[Tuple[int, int]]) -> float:
        """Calculate total distance of route in meters."""
        total_distance = 0.0

        for i in range(len(route) - 1):
            total_distance += self._calculate_distance(route[i], route[i + 1])

        return total_distance * 10  # Convert to meters

    def _calculate_distance(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate distance between points.

        Args:
            pos1: First position (x, y)
            pos2: Second position (x, y)

        Returns:
            Distance in cells
        """
        x1, y1 = pos1
        x2, y2 = pos2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def _is_at_position(self,
                        current: Tuple[int, int],
                        target: Tuple[int, int],
                        threshold: int) -> bool:
        """
        Check if current position is within threshold of target.

        Args:
            current: Current position
            target: Target position
            threshold: Maximum distance to consider "at position"

        Returns:
            Boolean indicating if positions are within threshold
        """
        return self._calculate_distance(current, target) <= threshold

    def _identify_coordination_points(self,
                                      split_points: List[Tuple[int, int]],
                                      support_positions: List[TacticalPosition],
                                      assault_positions: List[TacticalPosition]) -> List[CoordinationPoint]:
        """
        Identify required coordination points for movement.

        Args:
            split_points: List of potential split points
            support_positions: List of support positions
            assault_positions: List of assault positions

        Returns:
            List of coordination points with requirements
        """
        coordination_points = []

        # Add split point coordination
        if split_points:
            coordination_points.append(CoordinationPoint(
                position=split_points[0],
                coordination_type='split',
                required_elements={SquadElement.FULL},
                conditions=["squad_intact"],
                priority=1
            ))

        # Add support position coordination
        if support_positions:
            best_support = max(support_positions, key=lambda p: p.quality_score)
            coordination_points.append(CoordinationPoint(
                position=best_support.position,
                coordination_type='support_set',
                required_elements={SquadElement.SUPPORT},
                conditions=["clear_los", "weapons_ready"],
                priority=2
            ))

        # Add phase line before assault positions
        if assault_positions:
            # Calculate phase line position
            phase_x = sum(p.position[0] for p in assault_positions) // len(assault_positions)
            phase_y = sum(p.position[1] for p in assault_positions) // len(assault_positions)
            phase_line = (phase_x - 10, phase_y)  # 10 cells before assault positions

            coordination_points.append(CoordinationPoint(
                position=phase_line,
                coordination_type='phase_line',
                required_elements={SquadElement.ASSAULT, SquadElement.SUPPORT},
                conditions=["support_ready", "assault_ready"],
                priority=3
            ))

            # Add assault start point
            best_assault = max(assault_positions, key=lambda p: p.quality_score)
            coordination_points.append(CoordinationPoint(
                position=best_assault.position,
                coordination_type='assault_start',
                required_elements={SquadElement.ASSAULT},
                conditions=["support_set", "enemy_suppressed"],
                priority=4
            ))

        return coordination_points

    def _analyze_segment(self, path: List[Tuple[int, int]], movement_technique: str) -> RouteSegment:
        """
        Analyze characteristics of a route segment.

        Args:
            path: List of positions in segment
            movement_technique: Type of movement used

        Returns:
            RouteSegment with analyzed characteristics
        """
        if not path:
            return None

        # Initialize accumulators
        total_cover = 0.0
        total_concealment = 0.0
        total_threat = 0.0
        terrain_types = []

        # Analyze each position in path
        for x, y in path:
            terrain_info = self.terrain[y][x]
            terrain_types.append(terrain_info.terrain_type)

            # Calculate protection values
            total_cover += terrain_info.cover_bonus
            total_concealment += (1 - terrain_info.visibility_factor)
            total_threat += self.threat_matrix[y][x]

        # Calculate averages
        path_length = len(path)
        avg_cover = total_cover / path_length
        avg_concealment = total_concealment / path_length
        threat_exposure = total_threat / path_length

        return RouteSegment(
            start_pos=path[0],
            end_pos=path[-1],
            path=path,
            movement_technique=movement_technique,
            terrain_types=terrain_types,
            avg_cover=avg_cover,
            avg_concealment=avg_concealment,
            threat_exposure=threat_exposure
        )

    def _determine_movement_technique(
            self,
            start: Tuple[int, int],
            end: Tuple[int, int],
            element_type: SquadElement
    ) -> str:
        """
        Determine appropriate movement technique based on tactical factors.

        Args:
            start: Start position
            end: End position
            element_type: Type of element moving

        Returns:
            'bounding' or 'traveling' movement technique
        """
        # Get average threat level along direct path
        points = self._get_line_points(start[0], start[1], end[0], end[1])
        avg_threat = np.mean([self.threat_matrix[y][x] for x, y in points])

        # Get terrain characteristics
        terrain_exposure = np.mean([
            self.terrain[y][x].visibility_factor
            for x, y in points
        ])

        # Check distance
        distance = self._calculate_distance(start, end)

        # Decision factors for bounding movement:
        use_bounding = (
            # High threat level
                avg_threat > 0.6 or

                # High exposure with significant distance
                (terrain_exposure > 0.7 and distance > 20) or

                # Assault element in final approach
                (element_type == SquadElement.ASSAULT and distance > 15) or

                # Support element moving between positions
                (element_type == SquadElement.SUPPORT and avg_threat > 0.4) or

                # Any element in very high threat
                any(self.threat_matrix[y][x] > 0.8 for x, y in points)
        )

        return 'bounding' if use_bounding else 'traveling'

    def _estimate_movement_time(self, segments: List[RouteSegment]) -> float:
        """
        Estimate total movement time in minutes.

        Args:
            segments: List of route segments

        Returns:
            Estimated movement time in minutes
        """
        total_time = 0.0

        for segment in segments:
            # Base movement time from terrain and distance
            terrain_speeds = [self.movement_speeds[t] for t in segment.terrain_types]
            base_time = len(segment.path) / np.mean(terrain_speeds)

            # Movement technique modifier
            if segment.movement_technique == 'bounding':
                base_time *= 2.5  # Bounding takes longer

            # Add time for elevation changes
            elevation_changes = self._count_elevation_changes(segment.path)
            total_time += base_time + (elevation_changes * 0.5)

        return total_time

    def _calculate_energy_expenditure(self, segments: List[RouteSegment]) -> float:
        """
        Calculate relative energy cost of route.

        Args:
            segments: List of route segments

        Returns:
            Relative energy expenditure value
        """
        total_energy = 0.0

        for segment in segments:
            # Base energy from terrain
            terrain_energy = sum(self.energy_costs[t] for t in segment.terrain_types)

            # Additional energy for elevation changes
            elevation_energy = self._count_elevation_changes(segment.path) * 2

            # Movement technique modifier
            technique_multiplier = 2.0 if segment.movement_technique == 'bounding' else 1.0

            segment_energy = (terrain_energy + elevation_energy) * technique_multiplier
            total_energy += segment_energy

        return total_energy

    def _find_tactical_path(
            self,
            start: Tuple[int, int],
            end: Tuple[int, int],
            strategy: RouteStrategy,
            element_type: SquadElement
    ) -> List[Tuple[int, int]]:
        """
        Find optimal tactical path using enhanced A* algorithm.

        Args:
            start: Starting position
            end: End position
            strategy: Route planning strategy
            element_type: Type of element moving

        Returns:
            List of positions forming the path
        """
        # Initialize open and closed sets
        open_set = []
        closed_set = set()
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, end)}

        heappush(open_set, (f_score[start], start))

        while open_set:
            current = heappop(open_set)[1]

            if current == end:
                return self._reconstruct_path(came_from, current)

            closed_set.add(current)

            # Get neighbors considering tactical factors
            for neighbor in self._get_tactical_neighbors(current, element_type):
                if neighbor in closed_set:
                    continue

                # Calculate cost based on strategy and element type
                tentative_g = g_score[current] + self._tactical_cost(
                    current,
                    neighbor,
                    strategy,
                    element_type
                )

                if (neighbor not in g_score or
                        tentative_g < g_score[neighbor]):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, end)
                    heappush(open_set, (f_score[neighbor], neighbor))

        return []  # No path found

    def _tactical_cost(
            self,
            current: Tuple[int, int],
            next_pos: Tuple[int, int],
            strategy: RouteStrategy,
            element_type: SquadElement
    ) -> float:
        """Calculate tactical movement cost between positions."""
        x, y = next_pos
        terrain_info = self.terrain[y][x]

        # Base movement cost
        base_cost = self.movement_speeds[terrain_info.terrain_type]

        # Elevation change cost
        elevation_cost = self._elevation_cost(current, next_pos)

        # Threat exposure cost
        threat_cost = self.threat_matrix[y][x]

        # Protection values
        cover = terrain_info.cover_bonus
        concealment = 1 - terrain_info.visibility_factor

        # Apply strategy-specific weights
        if strategy == RouteStrategy.SPEED:
            return base_cost + elevation_cost * 0.5

        elif strategy == RouteStrategy.COVER:
            if element_type == SquadElement.SUPPORT:
                # Support element prioritizes good cover positions
                cover_cost = 1 - cover
                return base_cost * 0.3 + cover_cost * 0.4 + threat_cost * 0.3

            else:
                # Others balance cover with movement
                cover_cost = 1 - cover
                return base_cost * 0.4 + cover_cost * 0.3 + threat_cost * 0.3

        elif strategy == RouteStrategy.CONCEALMENT:
            if element_type == SquadElement.ASSAULT:
                # Assault element prioritizes concealment
                return (base_cost * 0.3 +
                        concealment * 0.4 +
                        threat_cost * 0.3)
            else:
                # Others balance concealment with other factors
                return (base_cost * 0.3 +
                        concealment * 0.3 +
                        threat_cost * 0.4)

        else:  # BALANCED
            return (base_cost * 0.25 +
                    (1 - cover) * 0.25 +
                    concealment * 0.25 +
                    threat_cost * 0.25)

    def _get_tactical_neighbors(
            self,
            pos: Tuple[int, int],
            element_type: SquadElement
    ) -> List[Tuple[int, int]]:
        """Get valid neighboring positions considering tactical factors."""
        x, y = pos
        neighbors = []

        # Check all 8 directions
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0),
                       (1, 1), (-1, 1), (1, -1), (-1, -1)]:
            new_x, new_y = x + dx, y + dy

            # Check bounds
            if not (0 <= new_x < self.env_width and
                    0 <= new_y < self.env_height):
                continue

            # Get terrain info
            terrain_info = self.terrain[new_y][new_x]

            # Check if position is traversable based on element type
            if element_type == SquadElement.FULL:
                # Full squad needs more space
                if terrain_info.terrain_type == TerrainType.STRUCTURE:
                    continue
            else:
                # Individual elements can move through more restricted terrain
                if terrain_info.movement_cost > 2.5:
                    continue

            # Add valid neighbor
            neighbors.append((new_x, new_y))

        return neighbors

    def _count_elevation_changes(self, path: List[Tuple[int, int]]) -> int:
        """Count number of elevation changes along path."""
        if len(path) < 2:
            return 0

        changes = 0
        prev_elevation = self.terrain[path[0][1]][path[0][0]].elevation_type

        for x, y in path[1:]:
            curr_elevation = self.terrain[y][x].elevation_type
            if curr_elevation != prev_elevation:
                changes += 1
            prev_elevation = curr_elevation

        return changes

    def _get_line_points(self, x1: int, y1: int, x2: int, y2: int) -> List[Tuple[int, int]]:
        """Get points along a line using Bresenham's algorithm."""
        points = []
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1

        sx = 1 if x2 > x1 else -1
        sy = 1 if y2 > y1 else -1

        if dx > dy:
            err = dx / 2
            while x != x2:
                points.append((x, y))
                err -= dy
                if err < 0:
                    y += sy
                    err += dx
                x += sx
        else:
            err = dy / 2
            while y != y2:
                points.append((x, y))
                err -= dx
                if err < 0:
                    x += sx
                    err += dy
                y += sy

        points.append((x2, y2))
        return points

    def _heuristic(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate heuristic estimate for A* pathfinding using enhanced distance
        calculation that considers terrain and elevation.

        Args:
            pos1: First position (x1, y1)
            pos2: Second position (x2, y2)

        Returns:
            Float value representing estimated cost to goal
        """
        x1, y1 = pos1
        x2, y2 = pos2

        # Base Manhattan distance
        base_distance = abs(x2 - x1) + abs(y2 - y1)

        # Get elevation info for both positions
        elev1 = self.terrain[y1][x1].elevation_type
        elev2 = self.terrain[y2][x2].elevation_type

        # Add elevation penalty if moving to higher ground
        if elev2.value > elev1.value:
            base_distance *= 1.2

        # Add terrain-based multiplier for rough estimate
        terrain_info = self.terrain[y2][x2]
        terrain_multiplier = {
            TerrainType.BARE: 1.0,
            TerrainType.SPARSE_VEG: 1.1,
            TerrainType.DENSE_VEG: 1.2,
            TerrainType.WOODS: 1.3,
            TerrainType.STRUCTURE: 1.4
        }[terrain_info.terrain_type]

        # Add threat-based cost component
        threat_factor = 1.0 + (self.threat_matrix[y2][x2] * 0.5)

        return base_distance * terrain_multiplier * threat_factor

    def _reconstruct_path(self, came_from: Dict[Tuple[int, int], Tuple[int, int]],
                          current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """
        Reconstruct path from A* pathfinding results.

        Args:
            came_from: Dictionary mapping positions to their predecessors
            current: End position to reconstruct path to

        Returns:
            List of positions forming the complete path
        """
        path = [current]

        while current in came_from:
            current = came_from[current]
            path.append(current)

        # Verify path is continuous
        for i in range(len(path) - 1):
            if not self._are_adjacent(path[i], path[i + 1]):
                logging.warning(f"Path discontinuity detected between {path[i]} and {path[i + 1]}")

        return path[::-1]  # Reverse to get start-to-end order

    def _elevation_cost(self, current: Tuple[int, int], next_pos: Tuple[int, int]) -> float:
        """
        Calculate movement cost based on elevation changes.

        Args:
            current: Current position (x1, y1)
            next_pos: Next position (x2, y2)

        Returns:
            Float value representing elevation-based movement cost
        """
        # Get elevation types
        current_elev = self.terrain[current[1]][current[0]].elevation_type
        next_elev = self.terrain[next_pos[1]][next_pos[0]].elevation_type

        # Base cost for level movement
        if current_elev == next_elev:
            return 1.0

        # Cost multipliers for elevation changes
        elevation_multipliers = {
            # Moving to higher elevation
            (ElevationType.GROUND_LEVEL, ElevationType.ELEVATED_LEVEL): 2.0,
            (ElevationType.LOWER_LEVEL, ElevationType.GROUND_LEVEL): 1.8,
            (ElevationType.LOWER_LEVEL, ElevationType.ELEVATED_LEVEL): 2.5,

            # Moving to lower elevation
            (ElevationType.ELEVATED_LEVEL, ElevationType.GROUND_LEVEL): 1.3,
            (ElevationType.GROUND_LEVEL, ElevationType.LOWER_LEVEL): 1.2,
            (ElevationType.ELEVATED_LEVEL, ElevationType.LOWER_LEVEL): 1.5
        }

        return elevation_multipliers.get((current_elev, next_elev), 1.0)

    def _are_adjacent(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> bool:
        """
        Check if two positions are adjacent (including diagonals).

        Args:
            pos1: First position (x1, y1)
            pos2: Second position (x2, y2)

        Returns:
            Boolean indicating if positions are adjacent
        """
        x1, y1 = pos1
        x2, y2 = pos2

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # Adjacent if separation is at most 1 cell in any direction
        return dx <= 1 and dy <= 1 and (dx + dy) > 0

    def _get_elevation_gradient(self, pos1: Tuple[int, int], pos2: Tuple[int, int]) -> float:
        """
        Calculate elevation gradient between positions.
        Used by _elevation_cost for refined cost calculations.

        Args:
            pos1: First position (x1, y1)
            pos2: Second position (x2, y2)

        Returns:
            Float representing elevation gradient
        """
        elev1 = self.terrain[pos1[1]][pos1[0]].elevation_type.value
        elev2 = self.terrain[pos2[1]][pos2[0]].elevation_type.value

        # Calculate horizontal distance
        dx = abs(pos2[0] - pos1[0])
        dy = abs(pos2[1] - pos1[1])
        distance = max(1, math.sqrt(dx * dx + dy * dy))

        # Calculate elevation difference and gradient
        elev_diff = elev2 - elev1
        return abs(elev_diff) / distance

    def _validate_path_continuity(self, path: List[Tuple[int, int]]) -> bool:
        """
        Validate that a path is continuous with no gaps.
        Used by _reconstruct_path for verification.

        Args:
            path: List of positions forming a path

        Returns:
            Boolean indicating if path is continuous
        """
        if len(path) < 2:
            return True

        for i in range(len(path) - 1):
            if not self._are_adjacent(path[i], path[i + 1]):
                logging.error(f"Path discontinuity at index {i}: {path[i]} to {path[i + 1]}")
                return False

        return True

    def visualize_routes(self,
                         routes: Dict[SquadElement, TacticalRoute],
                         movement_timing: Dict[SquadElement, MovementTiming],
                         save_path: str = 'tactical_plan.png'):
        """
        Create comprehensive visualization of tactical movement plan.

        Args:
            routes: Dict mapping elements to their routes
            movement_timing: Dict mapping elements to timing info
            save_path: Path to save visualization
        """
        # Create figure with three subplots - map, coordination, and timeline
        fig = plt.figure(figsize=(20, 15))
        gs = plt.GridSpec(3, 2, height_ratios=[4, 4, 1])

        # Map view with routes
        ax_map = fig.add_subplot(gs[0:2, 0])
        self._plot_tactical_map(ax_map, routes)

        # Coordination view showing positions and areas
        ax_coord = fig.add_subplot(gs[0:2, 1])
        self._plot_coordination_view(ax_coord, routes)

        # Timeline view
        ax_time = fig.add_subplot(gs[2, :])
        self._plot_movement_timeline(ax_time, routes, movement_timing)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_tactical_map(self, ax: plt.Axes, routes: Dict[SquadElement, TacticalRoute]):
        """Plot tactical map with routes and key positions."""
        # Plot terrain background
        terrain_img = self._create_terrain_image()
        ax.imshow(terrain_img, extent=(0, self.env_width, 0, self.env_height))

        # Plot threat overlay
        threat_overlay = ax.imshow(
            self.threat_matrix,
            alpha=0.3,
            cmap='Reds',
            extent=(0, self.env_width, 0, self.env_height)
        )

        # Plot routes with phase distinctions
        route_colors = {
            SquadElement.FULL: 'white',
            SquadElement.SUPPORT: 'blue',
            SquadElement.ASSAULT: 'green'
        }

        for element, route in routes.items():
            color = route_colors[element]

            # Plot each segment with appropriate style
            for i, segment in enumerate(route.segments):
                path = segment.path
                x_coords = [p[0] for p in path]
                y_coords = [p[1] for p in path]

                # Different line styles for different movement techniques
                line_style = '--' if segment.movement_technique == 'bounding' else '-'

                # Plot route with arrows
                ax.plot(x_coords, y_coords,
                        color=color,
                        linestyle=line_style,
                        alpha=0.8,
                        linewidth=2,
                        label=f'{element.value} Route' if i == 0 else "")

                # Add direction arrows
                self._add_direction_arrows(ax, path, color, 0.8)

        # Plot coordination points
        self._plot_coordination_points(ax)

        # Plot position areas
        self._plot_position_areas(ax)

        ax.set_title('Tactical Movement Plan')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

    def _plot_coordination_view(self, ax: plt.Axes, routes: Dict[SquadElement, TacticalRoute]):
        """Plot coordination-focused view showing key positions and areas."""
        # Plot simplified terrain (just elevation contours)
        elevation_data = np.array([[self.terrain[y][x].elevation_type.value
                                    for x in range(self.env_width)]
                                   for y in range(self.env_height)])
        ax.contour(elevation_data, levels=3, colors='gray', alpha=0.5)

        # Plot key positions with influence areas
        for pos_type in [TacticalPositionType.FIRE_SUPPORT,
                         TacticalPositionType.SUPPORT_BY_FIRE,
                         TacticalPositionType.ASSAULT]:
            positions = [p for p in self.element_positions.support_positions
                         if p.position_type == pos_type]

            for pos in positions:
                self._plot_position_influence(ax, pos)

        # Plot coordination points with areas
        for cp in self.element_positions.coordination_points:
            self._plot_coordination_area(ax, cp)

        # Plot movement corridors
        self._plot_movement_corridors(ax, routes)

        ax.set_title('Coordination and Control Measures')
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.grid(True, alpha=0.3)

    def _plot_movement_timeline(self,
                                ax: plt.Axes,
                                routes: Dict[SquadElement, TacticalRoute],
                                movement_timing: Dict[SquadElement, MovementTiming]):
        """Plot timeline of movement phases and coordination."""
        element_colors = {
            SquadElement.FULL: 'white',
            SquadElement.SUPPORT: 'blue',
            SquadElement.ASSAULT: 'green'
        }

        # Set up timeline
        max_time = max(timing.end_time for timing in movement_timing.values())
        ax.set_xlim(0, max_time)
        ax.set_ylim(0, len(routes) * 2)

        # Plot element timelines
        for i, (element, timing) in enumerate(movement_timing.items()):
            y_pos = i * 2 + 1
            color = element_colors[element]

            # Plot phase bars
            current_time = timing.start_time
            for phase_name, duration in timing.phase_durations.items():
                ax.barh(y_pos, duration, left=current_time, height=0.3,
                        color=color, alpha=0.5, label=phase_name)
                current_time += duration

            # Plot coordination points
            for coord_type, coord_time in timing.coordination_times.items():
                ax.plot(coord_time, y_pos, 'ro')
                ax.text(coord_time, y_pos + 0.3, coord_type,
                        rotation=45, ha='right')

        # Customize timeline
        ax.set_yticks([i * 2 + 1 for i in range(len(routes))])
        ax.set_yticklabels([element.value for element in routes.keys()])
        ax.set_xlabel('Time (minutes)')
        ax.set_title('Movement Timeline')
        ax.grid(True, alpha=0.3)

    def _plot_coordination_points(self, ax: plt.Axes):
        """Plot coordination points with type indicators."""
        coord_markers = {
            'support_set': 'o',
            'split': 's',
            'phase_line': '^',
            'assault_start': 'D'
        }

        for cp in self.element_positions.coordination_points:
            x, y = cp.position
            marker = coord_markers.get(cp.coordination_type, 'o')
            ax.plot(x, y, marker, color='yellow', markersize=10,
                    label=f'{cp.coordination_type} Point')

            # Add coordination area indicator
            circle = Circle((x, y), 5, fill=False, color='yellow', alpha=0.3)
            ax.add_patch(circle)

    def _plot_position_areas(self, ax: plt.Axes):
        """Plot tactical position areas with coverage sectors."""
        colors = {
            TacticalPositionType.FIRE_SUPPORT: 'blue',
            TacticalPositionType.SUPPORT_BY_FIRE: 'cyan',
            TacticalPositionType.ASSAULT: 'green'
        }

        # Plot support positions with coverage sectors
        for pos_type in colors:
            positions = [p for p in self.element_positions.support_positions
                         if p.position_type == pos_type]

            for pos in positions:
                x, y = pos.position
                color = colors[pos_type]

                # Position area
                rect = Rectangle(
                    (x - pos.unit_size.value, y - pos.unit_size.value),
                    pos.unit_size.value * 2,
                    pos.unit_size.value * 2,
                    fill=False,
                    color=color,
                    alpha=0.8
                )
                ax.add_patch(rect)

                # Coverage arc if applicable
                if hasattr(pos, 'coverage_arc'):
                    start_angle, end_angle = pos.coverage_arc
                    wedge = Wedge(
                        (x, y),
                        pos.max_range,
                        start_angle,
                        end_angle,
                        alpha=0.2,
                        color=color
                    )
                    ax.add_patch(wedge)

    def _plot_position_influence(self, ax: plt.Axes, position: TacticalPosition):
        """Plot position with its area of influence."""
        x, y = position.position

        # Plot base position
        ax.plot(x, y, 'o', color='white', markersize=8)

        # Plot fields of fire
        if hasattr(position, 'coverage_arc'):
            start_angle, end_angle = position.coverage_arc
            max_range = position.max_range

            # Primary sector
            primary_wedge = Wedge(
                (x, y),
                max_range,
                start_angle,
                end_angle,
                alpha=0.2,
                color='green'
            )
            ax.add_patch(primary_wedge)

            # Secondary sectors
            secondary_range = max_range * 0.7
            left_wedge = Wedge(
                (x, y),
                secondary_range,
                start_angle - 30,
                start_angle,
                alpha=0.1,
                color='yellow'
            )
            right_wedge = Wedge(
                (x, y),
                secondary_range,
                end_angle,
                end_angle + 30,
                alpha=0.1,
                color='yellow'
            )
            ax.add_patch(left_wedge)
            ax.add_patch(right_wedge)

    def _plot_coordination_area(self, ax: plt.Axes, coord_point: CoordinationPoint):
        """Plot coordination point with its control measures."""
        x, y = coord_point.position

        if coord_point.coordination_type == 'phase_line':
            # Draw phase line
            ax.axvline(x=x, color='yellow', linestyle='--', alpha=0.5)

        elif coord_point.coordination_type == 'split':
            # Draw split coordination area
            circle = Circle((x, y), 10, fill=False, color='white', alpha=0.5)
            ax.add_patch(circle)

        else:
            # Draw coordination point with required area
            ax.plot(x, y, 'o', color='yellow', markersize=8)
            circle = Circle((x, y), 5, fill=False, color='yellow', alpha=0.3)
            ax.add_patch(circle)

    def _plot_movement_corridors(self, ax: plt.Axes, routes: Dict[SquadElement, TacticalRoute]):
        """Plot movement corridors with control measures."""
        for element, route in routes.items():
            # Get route points
            points = []
            for segment in route.segments:
                points.extend(segment.path)

            if not points:
                continue

            # Calculate corridor width
            corridor_width = 20 if element == SquadElement.FULL else 10

            # Create corridor polygon
            left_bound = []
            right_bound = []

            for i in range(len(points) - 1):
                p1 = points[i]
                p2 = points[i + 1]

                # Calculate perpendicular vector
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    nx = -dy / length * corridor_width / 2
                    ny = dx / length * corridor_width / 2

                    # Add corridor bounds
                    left_bound.append((p1[0] + nx, p1[1] + ny))
                    right_bound.append((p1[0] - nx, p1[1] - ny))

            # Complete corridor polygon
            corridor = np.array(left_bound + right_bound[::-1])

            # Plot corridor
            ax.fill(corridor[:, 0], corridor[:, 1], alpha=0.1, color='white')

    def _create_terrain_image(self) -> np.ndarray:
        """Create RGB image representation of terrain."""
        terrain_img = np.zeros((self.env_height, self.env_width, 3))

        # Enhanced color mapping
        color_map = {
            TerrainType.BARE: [0.9, 0.9, 0.9],  # Light gray
            TerrainType.SPARSE_VEG: [0.8, 0.9, 0.8],  # Light green
            TerrainType.DENSE_VEG: [0.4, 0.7, 0.4],  # Medium green
            TerrainType.WOODS: [0.2, 0.5, 0.2],  # Dark green
            TerrainType.STRUCTURE: [0.6, 0.6, 0.7]  # Blue-gray
        }

        for y in range(self.env_height):
            for x in range(len(self.terrain[0])):
                terrain_info = self.terrain[y][x]
                base_color = color_map[terrain_info.terrain_type]

                # Apply elevation shading
                if terrain_info.elevation_type == ElevationType.ELEVATED_LEVEL:
                    # Lighter for elevated terrain
                    terrain_img[y, x] = [min(1.0, c * 1.2) for c in base_color]
                elif terrain_info.elevation_type == ElevationType.LOWER_LEVEL:
                    # Darker for depressions
                    terrain_img[y, x] = [c * 0.8 for c in base_color]
                else:
                    terrain_img[y, x] = base_color

        return terrain_img

    def _add_direction_arrows(self,
                              ax: plt.Axes,
                              path: List[Tuple[int, int]],
                              color: str,
                              alpha: float):
        """Add direction arrows to route visualization."""
        if len(path) < 2:
            return

        # Add arrows at regular intervals
        interval = max(len(path) // 5, 1)  # At least 5 arrows per path

        for i in range(0, len(path) - 1, interval):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dx = x2 - x1
            dy = y2 - y1

            # Calculate arrow position
            arrow_x = x1 + dx * 0.5
            arrow_y = y1 + dy * 0.5

            # Add arrow
            ax.arrow(arrow_x, arrow_y, dx * 0.1, dy * 0.1,
                     head_width=0.5,
                     head_length=0.8,
                     fc=color,
                     ec=color,
                     alpha=alpha)

    def visualize_route_comparison(self,
                                   routes: Dict[RouteStrategy, Dict[SquadElement, TacticalRoute]],
                                   save_path: str = 'route_comparison.png'):
        """
        Create visualization comparing routes from different strategies.

        Args:
            routes: Dict mapping strategies to their element routes
            save_path: Path to save visualization
        """
        fig, axes = plt.subplots(2, 2, figsize=(20, 20))
        fig.suptitle('Route Strategy Comparison', fontsize=16)

        for (strategy, element_routes), ax in zip(routes.items(), axes.flat):
            # Plot terrain and threat background
            terrain_img = self._create_terrain_image()
            ax.imshow(terrain_img, extent=(0, self.env_width, 0, self.env_height))

            threat_overlay = ax.imshow(
                self.threat_matrix,
                alpha=0.3,
                cmap='Reds',
                extent=(0, self.env_width, 0, self.env_height)
            )

            # Plot routes for each element
            for element, route in element_routes.items():
                self._plot_route_with_metrics(ax, route, element)

            ax.set_title(f'{strategy.value} Strategy')
            ax.set_xlabel('X Coordinate')
            ax.set_ylabel('Y Coordinate')
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_route_with_metrics(self,
                                 ax: plt.Axes,
                                 route: TacticalRoute,
                                 element: SquadElement):
        """Plot route with metric indicators."""
        element_colors = {
            SquadElement.FULL: 'white',
            SquadElement.SUPPORT: 'blue',
            SquadElement.ASSAULT: 'green'
        }
        color = element_colors[element]

        # Plot segments
        for segment in route.segments:
            path = segment.path
            x_coords = [p[0] for p in path]
            y_coords = [p[1] for p in path]

            # Line style based on movement technique
            line_style = '--' if segment.movement_technique == 'bounding' else '-'

            # Width based on quality score
            line_width = 1 + (segment.avg_cover + segment.avg_concealment) * 2

            # Plot route
            ax.plot(x_coords, y_coords,
                    color=color,
                    linestyle=line_style,
                    alpha=0.8,
                    linewidth=line_width,
                    label=f'{element.value} (Quality: {route.quality_score:.2f})')

            # Add metric indicators at midpoint
            mid_idx = len(path) // 2
            mid_point = path[mid_idx]

            # Add small indicators for key metrics
            self._add_metric_indicators(
                ax,
                mid_point,
                {
                    'Cover': segment.avg_cover,
                    'Concealment': segment.avg_concealment,
                    'Threat': segment.threat_exposure
                },
                color
            )

    def _add_metric_indicators(self,
                               ax: plt.Axes,
                               position: Tuple[int, int],
                               metrics: Dict[str, float],
                               color: str):
        """Add small visual indicators for route metrics."""
        x, y = position
        offset = 0

        for name, value in metrics.items():
            # Create small bar indicators
            bar_length = value * 2  # Scale for visibility
            ax.plot([x, x + bar_length], [y + offset, y + offset],
                    color=color, alpha=0.5, linewidth=1)
            ax.text(x + bar_length + 0.2, y + offset, name,
                    color=color, alpha=0.5, fontsize=6)
            offset += 0.5

    def save_analysis_report(self,
                             routes: Dict[SquadElement, TacticalRoute],
                             movement_timing: Dict[SquadElement, MovementTiming],
                             output_dir: str):
        """
        Save comprehensive analysis report with visualizations.

        Args:
            routes: Dict mapping elements to their routes
            movement_timing: Dict mapping elements to timing info
            output_dir: Directory to save report and visualizations
        """
        os.makedirs(output_dir, exist_ok=True)

        # Save main visualizations
        self.visualize_routes(
            routes=routes,
            movement_timing=movement_timing,
            save_path=os.path.join(output_dir, 'tactical_plan.png')
        )

        # Create text report
        report_path = os.path.join(output_dir, 'analysis_report.txt')
        with open(report_path, 'w') as f:
            f.write("Tactical Movement Analysis Report\n")
            f.write("================================\n\n")

            # Overall summary
            f.write("Overall Plan Summary:\n")
            f.write("-----------------\n")
            total_distance = sum(r.total_distance for r in routes.values())
            total_time = max(t.end_time for t in movement_timing.values())
            f.write(f"Total Distance: {total_distance:.0f}m\n")
            f.write(f"Total Time: {total_time:.1f} minutes\n\n")

            # Element details
            for element, route in routes.items():
                f.write(f"\n{element.value} Element Analysis:\n")
                f.write("-" * (len(element.value) + 19) + "\n")

                # Route characteristics
                f.write(f"Distance: {route.total_distance:.0f}m\n")
                f.write(f"Movement Time: {route.movement_time_estimate:.1f} minutes\n")
                f.write(f"Quality Score: {route.quality_score:.2f}\n")
                f.write(f"Average Cover: {route.avg_cover:.2f}\n")
                f.write(f"Average Concealment: {route.avg_concealment:.2f}\n")
                f.write(f"Threat Exposure: {route.total_threat_exposure:.2f}\n")

                # Timing details
                timing = movement_timing[element]
                f.write("\nMovement Phases:\n")
                for phase, duration in timing.phase_durations.items():
                    f.write(f"- {phase}: {duration:.1f} minutes\n")

                f.write("\nCoordination Points:\n")
                for point, time in timing.coordination_times.items():
                    f.write(f"- {point}: H+{time:.1f}\n")

                f.write("\n")

        logging.info(f"Analysis report saved to {output_dir}")


# Test code
if __name__ == "__main__":
    import csv
    print("Testing tactical route planning...")

    def load_test_map(project_root: str, map_filename: str) -> List[List[TerrainInfo]]:
        """Load map from CSV file into TerrainInfo grid."""
        map_path = os.path.join(project_root, map_filename)
        print(f"Loading map from: {map_path}")

        if not os.path.exists(map_path):
            print(f"Error: Map file not found at {map_path}")
            return []

        terrain_grid = []

        try:
            with open(map_path, 'r') as f:
                reader = csv.DictReader(f)
                current_row = []
                current_y = 0

                for csv_row in reader:
                    try:
                        x = int(csv_row['x'])
                        y = int(csv_row['y'])
                        terrain_type_str = str(csv_row['terrain_type'])
                        elevation_type_str = str(csv_row['elevation_type'])

                        if y > current_y:
                            if current_row:
                                terrain_grid.append(current_row)
                            current_row = []
                            current_y = y

                        terrain_info = TerrainInfo(
                            terrain_type=TerrainType[terrain_type_str],
                            elevation_type=ElevationType[elevation_type_str],
                            movement_cost=1.0,  # Set based on terrain type
                            visibility_factor=0.8 if TerrainType[terrain_type_str] in
                                                     [TerrainType.WOODS, TerrainType.DENSE_VEG] else 0.2,
                            cover_bonus=0.8 if TerrainType[terrain_type_str] in
                                               [TerrainType.STRUCTURE, TerrainType.WOODS] else 0.2
                        )

                        current_row.append(terrain_info)

                    except (KeyError, ValueError) as e:
                        print(f"Error processing row: {csv_row}")
                        print(f"Error details: {str(e)}")
                        continue

                if current_row:
                    terrain_grid.append(current_row)

        except Exception as e:
            print(f"Error loading map: {str(e)}")
            import traceback
            traceback.print_exc()
            return []

        return terrain_grid

    def test_tactical_route_planning():
        """Test integrated tactical route planning and visualization."""
        print("\nTesting Tactical Route Planning...")

        # Load test map
        map_filename = 'generated_map.csv'
        terrain = load_test_map(project_root, map_filename)

        if not terrain:
            print("Failed to load terrain map!")
            return

        env_height = len(terrain)
        env_width = len(terrain[0])

        # Set test parameters
        start_pos = (50, 50)  # Starting position
        objective = (350, 50)  # Objective position

        print(f"Planning routes from {start_pos} to objective {objective}")

        # Initialize route analyzer with position analyzer
        analyzer = TacticalRouteAnalyzer(terrain, env_width, env_height)
        analyzer.initialize_position_analyzer(objective)

        print("\nAnalyzing squad positions...")
        try:
            # Analyze positions
            element_positions = analyzer.analyze_squad_positions(objective)

            print(f"Found positions:")
            print(f"Support positions: {len(element_positions.support_positions)}")
            print(f"Assault positions: {len(element_positions.assault_positions)}")
            print(f"Split points: {len(element_positions.split_points)}")
            print(f"Coordination points: {len(element_positions.coordination_points)}")

            # Test routes with different strategies
            for strategy in RouteStrategy:
                print(f"\nTesting {strategy.value} routes...")

                try:
                    # Find routes
                    routes = analyzer.find_tactical_routes(
                        start_pos=start_pos,
                        objective=objective,
                        strategy=strategy
                    )

                    print(f"Routes planned:")
                    for element, route in routes.items():
                        print(f"\n{element.value}:")
                        print(f"Total distance: {route.total_distance:.0f}m")
                        print(f"Movement time: {route.movement_time_estimate:.1f} minutes")
                        print(f"Quality score: {route.quality_score:.2f}")
                        print(f"Number of segments: {len(route.segments)}")

                        # Print segment details
                        for i, segment in enumerate(route.segments):
                            print(f"\nSegment {i + 1}:")
                            print(f"Length: {len(segment.path)} cells")
                            print(f"Movement technique: {segment.movement_technique}")
                            print(f"Average cover: {segment.avg_cover:.2f}")
                            print(f"Threat exposure: {segment.threat_exposure:.2f}")

                    # Get movement timing
                    movement_timing = analyzer.analyze_movement_timing(
                        routes,
                        element_positions.coordination_points
                    )

                    print("\nMovement Timing Analysis:")
                    for element, timing in movement_timing.items():
                        print(f"\n{element.value}:")
                        print(f"Start time: H+{timing.start_time:.1f}")
                        print(f"End time: H+{timing.end_time:.1f}")
                        print(f"Movement rate: {timing.movement_rate:.1f} m/min")

                        print("\nCoordination points:")
                        for point_type, time in timing.coordination_times.items():
                            print(f"- {point_type}: H+{time:.1f}")

                    # Create visualizations and save report
                    print("\nCreating visualizations...")
                    output_dir = f'analysis_output_{strategy.value}'
                    analyzer.save_analysis_report(
                        routes=routes,
                        movement_timing=movement_timing,
                        output_dir=output_dir
                    )

                    print(f"Analysis saved to {output_dir}/")

                except Exception as e:
                    print(f"Error planning routes for {strategy.value}: {str(e)}")
                    traceback.print_exc()

        except Exception as e:
            print(f"Error in position analysis: {str(e)}")
            traceback.print_exc()

    def test_individual_components():
        """Test individual components of route planning."""
        print("\nTesting individual components...")

        # Load test map
        map_filename = 'generated_map.csv'
        terrain = load_test_map(project_root, map_filename)

        if not terrain:
            print("Failed to load terrain map!")
            return

        env_height = len(terrain)
        env_width = len(terrain[0])

        # Test parameters
        start_pos = (50, 50)
        objective = (350, 50)

        # Initialize analyzers
        analyzer = TacticalRouteAnalyzer(terrain, env_width, env_height)
        analyzer.initialize_position_analyzer(objective)

        # Test position analysis
        print("\nTesting position analysis...")
        try:
            positions = analyzer.analyze_squad_positions(objective)

            print("\nSupport Positions:")
            for i, pos in enumerate(positions.support_positions[:3]):
                print(f"\nPosition {i + 1}:")
                print(f"Type: {pos.position_type.value}")
                print(f"Location: {pos.position}")
                print(f"Quality: {pos.quality_score:.2f}")
                print(f"Coverage arc: {pos.coverage_arc}")

            print("\nAssault Positions:")
            for i, pos in enumerate(positions.assault_positions[:3]):
                print(f"\nPosition {i + 1}:")
                print(f"Location: {pos.position}")
                print(f"Quality: {pos.quality_score:.2f}")

            print("\nSplit Points:")
            for i, point in enumerate(positions.split_points[:3]):
                print(f"Point {i + 1}: {point}")

        except Exception as e:
            print(f"Error in position analysis: {str(e)}")
            traceback.print_exc()

        # Test coordination point analysis
        print("\nTesting coordination point analysis...")
        try:
            coord_points = analyzer._identify_coordination_points(
                positions.split_points,
                positions.support_positions,
                positions.assault_positions
            )

            print("\nCoordination Points:")
            for i, cp in enumerate(coord_points):
                print(f"\nPoint {i + 1}:")
                print(f"Type: {cp.coordination_type}")
                print(f"Position: {cp.position}")
                print(f"Required elements: {[e.value for e in cp.required_elements]}")
                print(f"Priority: {cp.priority}")

        except Exception as e:
            print(f"Error in coordination analysis: {str(e)}")
            traceback.print_exc()

        # Test route segment analysis
        print("\nTesting route segment analysis...")
        try:
            # Create a test route segment
            test_route = [(50, 50), (55, 52), (60, 55), (65, 57)]
            segment = RouteSegment(
                start_pos=test_route[0],
                end_pos=test_route[-1],
                path=test_route,
                movement_technique='traveling',
                terrain_types=[terrain[y][x].terrain_type for x, y in test_route],
                avg_cover=0.7,
                avg_concealment=0.6,
                threat_exposure=0.3
            )

            # Test timing calculation
            timing = analyzer._calculate_phase_duration(
                test_route,
                SquadElement.SUPPORT,
                "initial_movement"
            )

            print(f"\nSegment timing: {timing:.1f} minutes")
            print(f"Average movement rate: {len(test_route) * 10 / timing:.1f} m/min")

        except Exception as e:
            print(f"Error in segment analysis: {str(e)}")
            traceback.print_exc()

    # Run the tests
    try:
        print("Starting Tactical Route Analyzer Tests...")

        print("\n=== Testing Full Route Planning ===")
        test_tactical_route_planning()

        # print("\n=== Testing Individual Components ===")
        # test_individual_components()

        print("\nAll tests completed!")

    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        traceback.print_exc()


