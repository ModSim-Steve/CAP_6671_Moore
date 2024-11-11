"""
Tactical Position Analyzer Module

Purpose:
    Analyzes and evaluates potential tactical positions based on military considerations,
    focusing on identifying and evaluating fire support positions, support by fire positions,
    and assault positions with emphasis on threat engagement capabilities.

Key Features:
    1. Position type classification and evaluation
       - Fire support positions
       - Support by fire positions
       - Assault positions
       - Threat engagement analysis

    2. Analysis capabilities:
       - Coverage arc calculation
       - Fields of fire evaluation
       - Cover and concealment assessment
       - Threat exposure analysis
       - Position quality scoring
       - Engagement quality assessment

    3. Support for military position requirements:
       - Minimum/maximum engagement ranges
       - Required observation arcs
       - Cover requirements
       - Elevation advantages
       - Threat engagement capabilities

Classes:
    UnitSize: Defines unit sizes (team/squad) and their requirements
    TacticalPositionType: Defines different types of tactical positions
    TacticalPosition: Represents an evaluated tactical position
    TacticalPositionAnalyzer: Main analysis class
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict, Set, Mapping, TypedDict, Literal, Union, Optional
import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import os
import sys
import csv

from Russian_AF_ASLT_DET_Cap_SQD import (
    Soldier,
    TeamMember,
    SpecialTeam,
    Team,
    Squad,
    AK12
)

from pathfinding import (
    TerrainInfo,
    TerrainType,
    ElevationType,
    PathCosts
)

# Add project root to Python path for imports
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)


@dataclass
class EnemyThreat:
    """Represents an enemy unit with its threat characteristics."""
    position: Tuple[int, int]  # (x, y) coordinates
    unit: 'Squad'  # Forward reference to Squad type
    observation_range: int  # Range in cells
    engagement_range: int  # Range in cells
    suspected_accuracy: float  # 0.0 to 1.0


class UnitSize(Enum):
    """
    Size of unit the position should accommodate.

    Attributes:
        TEAM: 4-person team size (7x7 cells)
        SQUAD: 9-person squad size (12x12 cells)

    Usage:
        Used to determine area requirements and formation spacing
        for tactical positions.
    """
    TEAM = "team"
    SQUAD = "squad"


class QualityFactor(TypedDict):
    weight: float
    min: Optional[float]
    max: Optional[float]


class PositionEvaluation(TypedDict):
    quality_score: float
    coverage_arc: Tuple[float, float]  # Explicitly tuple of two floats
    covered_threats: List[Tuple[int, int]]
    engagement_quality: float
    has_elevation: bool
    distance_to_objective: float


class EngagementRequirements(TypedDict):
    type: Literal['engagement']
    min_range: float
    max_range: float
    min_quality: float
    min_los_quality: float
    min_cover: Optional[float]
    min_concealment: Optional[float]
    max_threat: Optional[float]
    description: str


class AssaultRequirements(TypedDict):
    type: Literal['assault']
    min_range: Optional[float]
    max_range: float
    min_quality: Optional[float]
    min_los_quality: Optional[float]
    min_cover: float
    min_concealment: float
    max_threat: float
    description: str


PositionRequirements = TypedDict('PositionRequirements', {
    'primary': Union[EngagementRequirements, AssaultRequirements],
    'quality_factors': Dict[str, QualityFactor]
})


class TacticalPositionType(Enum):
    """
    Types of tactical positions with distinct military purposes and requirements.

    Types:
        FIRE_SUPPORT:
            - Direct fire support position
            - Requires good observation and fields of fire
            - Strong cover requirement
            - Can be within enemy engagement range if good cover exists
            - Prefers elevated terrain
            - Must be able to engage at least one enemy position

        SUPPORT_BY_FIRE:
            - Prioritizes staying outside enemy direct fire
            - Requires good cover and moderate concealment
            - Needs good fields of fire to objective
            - Balance between effectiveness and protection
            - Must be able to engage at least one enemy position

        ASSAULT:
            - Last covered and concealed position before assault
            - Must be close to objective (will be in enemy engagement range)
            - Prioritizes concealment over cover
            - Limited fields of fire needed
            - Route to objective more important than position protection
    """
    FIRE_SUPPORT = "fire_support"
    SUPPORT_BY_FIRE = "support_by_fire"
    ASSAULT = "assault"

    def get_requirements(self) -> PositionRequirements:
        """Get detailed requirements for position type."""
        if self == TacticalPositionType.FIRE_SUPPORT:
            return {
                'primary': {
                    'type': 'engagement',
                    'min_range': 600.0,
                    'max_range': 2000.0,
                    'min_quality': 0.3,
                    'min_los_quality': 0.5,
                    'min_cover': 0.0,  # Optional but included
                    'min_concealment': 0.0,  # Optional but included
                    'max_threat': 0.7,  # Optional but included
                    'description': 'Long-range position with good fields of fire'
                },
                'quality_factors': {
                    'los_quality': {'weight': 0.4, 'min': 0.5, 'max': None},
                    'elevation_advantage': {'weight': 0.3, 'min': 0.0, 'max': None},
                    'cover': {'weight': 0.2, 'min': 0.0, 'max': None},
                    'concealment': {'weight': 0.1, 'min': 0.0, 'max': None}
                }
            }
        elif self == TacticalPositionType.SUPPORT_BY_FIRE:
            return {
                'primary': {
                    'type': 'engagement',
                    'min_range': 400.0,
                    'max_range': 750.0,
                    'min_quality': 0.4,
                    'min_los_quality': 0.6,
                    'min_cover': 0.2,  # Optional but included
                    'min_concealment': 0.2,  # Optional but included
                    'max_threat': 0.5,  # Optional but included
                    'description': 'Medium-range position for direct fire support'
                },
                'quality_factors': {
                    'los_quality': {'weight': 0.4, 'min': 0.6, 'max': None},
                    'elevation_advantage': {'weight': 0.2, 'min': 0.0, 'max': None},
                    'cover': {'weight': 0.2, 'min': 0.2, 'max': None},
                    'concealment': {'weight': 0.2, 'min': 0.2, 'max': None}
                }
            }
        else:  # ASSAULT
            return {
                'primary': {
                    'type': 'assault',
                    'min_range': None,
                    'max_range': 500.0,
                    'min_quality': None,
                    'min_los_quality': None,
                    'min_cover': 0.6,
                    'min_concealment': 0.6,
                    'max_threat': 0.3,
                    'description': 'Final covered and concealed position before assault'
                },
                'quality_factors': {
                    'cover': {'weight': 0.4, 'min': 0.6, 'max': None},
                    'concealment': {'weight': 0.4, 'min': 0.6, 'max': None},
                    'threat_exposure': {'weight': 0.2, 'min': None, 'max': 0.3}
                }
            }


@dataclass
class TacticalPosition:
    """
    Represents an evaluated tactical position with its characteristics and capabilities.

    Attributes:
        position (Tuple[int, int]): Center position (x, y)
        cells (List[Tuple[int, int]]): All cells that make up the position
        position_type (TacticalPositionType): Type of tactical position
        unit_size (UnitSize): Size of unit position accommodates
        coverage_arc (Tuple[float, float]): Start and end angles of coverage arc
        max_range (int): Maximum engagement range
        quality_score (float): Overall position quality (0-1)
        covered_threats (List[Tuple[int, int]]): List of engage-able threat positions
        engagement_quality (float): Quality of possible engagements (0-1)
        has_elevation (bool): Whether position has elevation advantage
    """
    position: Tuple[int, int]
    cells: List[Tuple[int, int]]
    position_type: TacticalPositionType
    unit_size: UnitSize
    coverage_arc: Tuple[float, float]
    max_range: int
    quality_score: float
    covered_threats: List[Tuple[int, int]]
    engagement_quality: float = 0.0
    has_elevation: bool = False


@dataclass
class PositionAttempt:
    """
    Tracks attempted position evaluations for debugging and analysis.

    Attributes:
        position (Tuple[int, int]): Position evaluated
        score (float): Quality score achieved
        failure_reason (str): Reason for position failure
        averages (Dict[str, float]): Average characteristics
        engagement_stats (Dict[str, Union[float, bool]]): Engagement capabilities
        threats_covered (int): Number of threats that could be engaged
        engagement_arc (Tuple[float, float]): Coverage arc achieved
    """
    position: Tuple[int, int]
    score: float
    failure_reason: str
    averages: Dict[str, float]
    engagement_stats: Dict[str, Union[float, bool]] = field(default_factory=dict)
    threats_covered: int = 0
    engagement_arc: Tuple[float, float] = (0.0, 0.0)


class TacticalPositionAnalyzer:
    """
    Analyzes and evaluates potential tactical positions based on terrain, enemy threats,
    and unit requirements.

    Key Capabilities:
        1. Position identification for different tactical purposes
        2. Unit size-based position evaluation
        3. Threat exposure analysis
        4. Fields of fire calculation
        5. Position quality scoring
        6. Engagement capability analysis

    Attributes:
        terrain_analyzer: Reference to main terrain analyzer for threat data
        env_width (int): Environment width in cells
        env_height (int): Environment height in cells
        threat_matrix (np.ndarray): Matrix of threat values for each cell
        enemy_threats (List[Dict]): List of known/suspected enemy positions
        top_failures (List[PositionAttempt]): Tracks near-miss positions for debugging
    """

    def __init__(self, terrain_analyzer, env_width: int, env_height: int, objective: Tuple[int, int]):
        """
        Initialize the tactical position analyzer.

        Args:
            terrain_analyzer: Reference to main terrain analyzer for threat data
            env_width: Width of environment in cells
            env_height: Height of environment in cells
        """
        self.env_width = env_width
        self.env_height = env_height
        self.threat_matrix = np.zeros((env_height, env_width))
        self.terrain_analyzer = terrain_analyzer
        self.enemy_threats = []
        self.top_failures = []
        self.objective = objective
        print(f"Initialized TacticalPositionAnalyzer with objective at {objective}")

    def find_tactical_positions(self,
                                terrain: List[List[TerrainInfo]],
                                position_type: TacticalPositionType,
                                unit_size: UnitSize,
                                objective: Tuple[int, int],
                                min_range: int,
                                max_range: int) -> List[TacticalPosition]:
        """
        Find suitable tactical positions based on type and unit requirements.

        Args:
            terrain: 2D grid of TerrainInfo objects
            position_type: Type of tactical position to find
            unit_size: Size of unit to accommodate
            objective: Target position (x, y)
            min_range: Minimum range from objective in cells (1 cell = 10m)
            max_range: Maximum range from objective in cells

        Returns:
            List of TacticalPosition objects sorted by quality score
        """
        positions = []
        dimensions = self._get_unit_dimensions(unit_size)
        self.top_failures = []

        # Get requirements for position type
        requirements = position_type.get_requirements()
        primary_reqs = requirements['primary']

        # Convert requirements from meters to cells
        min_req_range = int((primary_reqs.get('min_range', 0) or 0) / 10)
        max_req_range = int(primary_reqs['max_range'] / 10)

        # Use most restrictive range requirements
        search_min_range = max(min_range, min_req_range)
        search_max_range = min(max_range, max_req_range)

        print(f"\n=== Searching for {unit_size.value}-sized {position_type.value} Positions ===")
        print(f"Required area: {dimensions}x{dimensions} cells")
        print(
            f"Range requirements: {search_min_range}-{search_max_range} cells ({search_min_range * 10}-{search_max_range * 10}m)")
        print(f"Objective location: {objective}")

        # Get valid range ring cells for efficiency
        valid_cells = self._get_valid_range_cells(objective, search_min_range, search_max_range)
        print(f"Valid cells in range ring: {len(valid_cells)}")

        positions_evaluated = 0
        positions_found = 0

        # Search through valid cells in unit-sized steps
        for base_y in range(0, self.env_height - dimensions + 1, dimensions):
            for base_x in range(0, self.env_width - dimensions + 1, dimensions):
                positions_evaluated += 1

                if positions_evaluated % 100 == 0:
                    print(f"Progress: Checked {positions_evaluated} potential positions...")

                # Quick check if any part of area is in valid range
                area_valid = False
                for dy in range(dimensions):
                    for dx in range(dimensions):
                        if (base_x + dx, base_y + dy) in valid_cells:
                            area_valid = True
                            break
                    if area_valid:
                        break

                if not area_valid:
                    continue

                # Initial position check
                if not self._is_suitable_position(base_x, base_y, terrain, position_type):
                    continue

                # Collect and validate area cells
                area_cells = self._get_area_cells(base_x, base_y, dimensions)

                # Skip if any cells are out of bounds
                if not all(0 <= x < self.env_width and 0 <= y < self.env_height
                           for x, y in area_cells):
                    continue

                # Validate entire area
                area_result = self._validate_unit_area(area_cells, terrain, position_type, objective)
                if not area_result['is_valid']:
                    continue

                positions_found += 1

                # Calculate position characteristics and create TacticalPosition object
                position_eval = self._evaluate_position(
                    area_cells,
                    terrain,
                    position_type,
                    objective
                )

                # Create tactical position with explicit typing
                center = (base_x + dimensions // 2, base_y + dimensions // 2)
                coverage_arc: Tuple[float, float] = position_eval['coverage_arc']
                new_position = TacticalPosition(
                    position=center,
                    cells=area_cells,
                    position_type=position_type,
                    unit_size=unit_size,
                    coverage_arc=coverage_arc,
                    max_range=search_max_range * 10,
                    quality_score=float(position_eval['quality_score']),
                    covered_threats=position_eval['covered_threats'],
                    engagement_quality=float(position_eval['engagement_quality']),
                    has_elevation=bool(position_eval['has_elevation'])
                )
                positions.append(new_position)

        print("\nAnalysis complete:")
        print(f"Total positions checked: {positions_evaluated}")
        print(f"Suitable positions found: {len(positions)}")

        # Report failures if no positions found
        if len(positions) == 0:
            print("\nTop 10 positions that almost qualified:")
            for i, attempt in enumerate(self.top_failures[:10], 1):
                self._print_failure_details(attempt, i)

        return sorted(positions, key=lambda p: p.quality_score, reverse=True)

    def _get_area_cells(self, base_x: int, base_y: int, dimensions: int) -> List[Tuple[int, int]]:
        """
        Get all cells in a unit-sized area.

        Args:
            base_x: Base x coordinate
            base_y: Base y coordinate
            dimensions: Size of area

        Returns:
            List of (x, y) coordinates for all cells in area
        """
        return [(base_x + dx, base_y + dy)
                for dy in range(dimensions)
                for dx in range(dimensions)]

    def _get_valid_range_cells(self, objective: Tuple[int, int],
                               min_range: int, max_range: int) -> Set[Tuple[int, int]]:
        """
        Get all valid cells within the range ring from objective point.
        Uses set for faster membership testing.

        Args:
            objective: Center point (x, y)
            min_range: Minimum range from objective
            max_range: Maximum range from objective

        Returns:
            Set of (x, y) tuples representing valid cell positions
        """
        valid_cells = set()
        obj_x, obj_y = objective

        # Calculate bounds to minimize unnecessary checks
        min_x = max(0, obj_x - max_range)
        max_x = min(self.env_width, obj_x + max_range + 1)
        min_y = max(0, obj_y - max_range)
        max_y = min(self.env_height, obj_y + max_range + 1)

        # Check cells within bounded box
        for y in range(min_y, max_y):
            for x in range(min_x, max_x):
                distance = math.sqrt((x - obj_x) ** 2 + (y - obj_y) ** 2)
                if min_range <= distance <= max_range:
                    valid_cells.add((x, y))

        return valid_cells

    def _get_unit_dimensions(self, unit_size: UnitSize) -> int:
        """
        Get required dimensions for unit size.

        Args:
            unit_size: Size of unit requiring position

        Returns:
            Integer representing dimension of square area needed
        """
        return 7 if unit_size == UnitSize.TEAM else 12

    def _is_suitable_position(self, x: int, y: int,
                              terrain: List[List[TerrainInfo]],
                              position_type: TacticalPositionType) -> bool:
        """
        Comprehensive position suitability check considering terrain, threats, and engagement capabilities.

        Args:
            x: X coordinate to check
            y: Y coordinate to check
            terrain: Terrain grid
            position_type: Type of tactical position being evaluated

        Returns:
            Boolean indicating if position is suitable
        """
        # Check bounds
        if not (0 <= x < self.env_width and 0 <= y < self.env_height):
            return False

        terrain_info = terrain[y][x]
        threat_level = float(self.threat_matrix[y][x])
        requirements = position_type.get_requirements()
        primary_reqs = requirements['primary']

        # Calculate distance to objective
        dist_to_objective = math.sqrt(
            (x - self.objective[0]) ** 2 +
            (y - self.objective[1]) ** 2
        ) * 10  # Convert to meters

        # Calculate base position stats for failure tracking
        position_stats = {
            'distance': float(dist_to_objective),
            'cover': float(terrain_info.cover_bonus),
            'concealment': float(1 - terrain_info.visibility_factor),
            'threat': float(threat_level),
            'elevation': float(terrain_info.elevation_type.value),
            'los_quality': 0.0,
            'engagement_quality': 0.0
        }

        # Check movement cost (applies to all position types)
        if terrain_info.movement_cost > 2.5:
            self._track_failure(
                position=(x, y),
                score=0.1,
                reason=f"Terrain too difficult to move through: {terrain_info.movement_cost:.1f}",
                averages=position_stats
            )
            return False

        # Handle engagement positions (Fire Support and Support by Fire)
        if primary_reqs['type'] == 'engagement':
            # Check range requirements
            min_range = float(primary_reqs['min_range'])
            max_range = float(primary_reqs['max_range'])

            if not (min_range <= dist_to_objective <= max_range):
                self._track_failure(
                    position=(x, y),
                    score=0.2,
                    reason=f"Out of engagement range: {dist_to_objective:.0f}m",
                    averages=position_stats
                )
                return False

            # Check line of sight to objective
            los_result = self._check_line_of_sight_quality((x, y), self.objective, terrain)
            position_stats['los_quality'] = float(los_result['los_quality'])

            if not los_result['has_los']:
                self._track_failure(
                    position=(x, y),
                    score=0.3,
                    reason=f"No line of sight to objective",
                    averages=position_stats
                )
                return False

            if los_result['los_quality'] < primary_reqs['min_los_quality']:
                self._track_failure(
                    position=(x, y),
                    score=0.35,
                    reason=f"Poor line of sight quality: {los_result['los_quality']:.2f}",
                    averages=position_stats
                )
                return False

            # Check engagement capability
            engagement = self._evaluate_engagement((x, y), self.objective, terrain)
            position_stats['engagement_quality'] = float(engagement['quality_score'])

            if not engagement['can_engage']:
                self._track_failure(
                    position=(x, y),
                    score=0.4,
                    reason="Cannot effectively engage objective",
                    averages=position_stats
                )
                return False

            # Check if we can establish a valid engagement arc
            arc = self._find_best_engagement_arc(
                (x, y),
                self._calculate_target_angle((x, y), self.objective),
                terrain,
                self.objective
            )
            if arc is None:
                self._track_failure(
                    position=(x, y),
                    score=0.45,
                    reason="Cannot establish valid engagement arc",
                    averages=position_stats
                )
                return False

            # Check minimum cover/concealment if specified
            if primary_reqs.get('min_cover') and terrain_info.cover_bonus < primary_reqs['min_cover']:
                self._track_failure(
                    position=(x, y),
                    score=0.5,
                    reason=f"Insufficient cover: {terrain_info.cover_bonus:.2f}",
                    averages=position_stats
                )
                return False

            if primary_reqs.get('min_concealment') and (1 - terrain_info.visibility_factor) < primary_reqs[
                'min_concealment']:
                self._track_failure(
                    position=(x, y),
                    score=0.5,
                    reason=f"Insufficient concealment: {1 - terrain_info.visibility_factor:.2f}",
                    averages=position_stats
                )
                return False

            return True

        # Handle assault positions
        else:  # primary_reqs['type'] == 'assault'
            # Check range to objective
            if dist_to_objective > primary_reqs['max_range']:
                self._track_failure(
                    position=(x, y),
                    score=0.2,
                    reason=f"Too far from objective: {dist_to_objective:.0f}m",
                    averages=position_stats
                )
                return False

            # Check minimum cover requirement
            if terrain_info.cover_bonus < primary_reqs['min_cover']:
                self._track_failure(
                    position=(x, y),
                    score=0.3,
                    reason=f"Insufficient cover: {terrain_info.cover_bonus:.2f}",
                    averages=position_stats
                )
                return False

            # Check minimum concealment requirement
            concealment = 1 - terrain_info.visibility_factor
            if concealment < primary_reqs['min_concealment']:
                self._track_failure(
                    position=(x, y),
                    score=0.3,
                    reason=f"Insufficient concealment: {concealment:.2f}",
                    averages=position_stats
                )
                return False

            # Check threat exposure
            if threat_level > primary_reqs['max_threat']:
                self._track_failure(
                    position=(x, y),
                    score=0.4,
                    reason=f"Too exposed to threat: {threat_level:.2f}",
                    averages=position_stats
                )
                return False

            return True

    def _calculate_position_score(self, terrain_info, threat_level,
                                  engagement_result, distance) -> float:
        """Calculate position quality score for ranking purposes."""
        # Base score from engagement capability
        if not engagement_result['can_engage']:
            return 0.0

        # Calculate range score
        if distance <= 500:
            range_score = 1.0
        elif distance <= 1000:
            range_score = 0.8
        elif distance <= 1500:
            range_score = 0.6
        else:
            range_score = 0.4

        # Calculate final score with elevation and cover as bonuses
        score = (
                engagement_result['quality_score'] * 0.4 +
                range_score * 0.3 +
                terrain_info.cover_bonus * 0.15 +
                (0.15 if terrain_info.elevation_type.value > 0 else 0.0) +
                (1 - threat_level) * 0.15
        )

        return score

    def _validate_unit_area(self,
                            area_cells: List[Tuple[int, int]],
                            terrain: List[List[TerrainInfo]],
                            position_type: TacticalPositionType,
                            objective: Tuple[int, int]) -> Dict[str, Union[bool, Dict[str, float]]]:
        """
        Validate that an entire unit-sized area is suitable.
        Performs more detailed analysis of area characteristics.

        Args:
            area_cells: List of cell coordinates in the area
            terrain: Terrain grid
            position_type: Type of tactical position
            objective: Target position (x, y)

        Returns:
            Dictionary containing:
            - is_valid: Boolean indicating if area is valid
            - stats: Dictionary of area statistics
            - reason: String explaining failure reason if invalid
        """
        # Calculate center position
        center_x = sum(x for x, _ in area_cells) // len(area_cells)
        center_y = sum(y for _, y in area_cells) // len(area_cells)
        center = (center_x, center_y)

        # Get position requirements
        requirements = position_type.get_requirements()
        primary_reqs = requirements['primary']

        # Calculate area characteristics
        area_stats = self._calculate_area_characteristics(area_cells, terrain)

        # Check if movement through area is feasible
        if area_stats['avg_movement_cost'] > 2.5:
            return {
                'is_valid': False,
                'stats': area_stats,
                'reason': f"Area average movement cost too high: {area_stats['avg_movement_cost']:.1f}"
            }

        # Handle engagement positions
        if primary_reqs['type'] == 'engagement':
            # Enhanced line of sight check from center
            los_result = self._check_line_of_sight_quality(center, objective, terrain)
            if not los_result['has_los']:
                return {
                    'is_valid': False,
                    'stats': area_stats,
                    'reason': "No line of sight from area center"
                }

            # Check line of sight quality requirement
            if los_result['los_quality'] < primary_reqs['min_los_quality']:
                return {
                    'is_valid': False,
                    'stats': area_stats,
                    'reason': f"Insufficient line of sight quality: {los_result['los_quality']:.2f}"
                }

            # Check minimum cover/concealment if specified
            if primary_reqs.get('min_cover') and area_stats['avg_cover'] < primary_reqs['min_cover']:
                return {
                    'is_valid': False,
                    'stats': area_stats,
                    'reason': f"Insufficient average cover: {area_stats['avg_cover']:.2f}"
                }

            if primary_reqs.get('min_concealment') and area_stats['avg_concealment'] < primary_reqs['min_concealment']:
                return {
                    'is_valid': False,
                    'stats': area_stats,
                    'reason': f"Insufficient average concealment: {area_stats['avg_concealment']:.2f}"
                }

        # Handle assault positions
        else:  # primary_reqs['type'] == 'assault'
            # Check minimum cover requirement
            if area_stats['avg_cover'] < primary_reqs['min_cover']:
                return {
                    'is_valid': False,
                    'stats': area_stats,
                    'reason': f"Insufficient average cover: {area_stats['avg_cover']:.2f}"
                }

            # Check minimum concealment requirement
            if area_stats['avg_concealment'] < primary_reqs['min_concealment']:
                return {
                    'is_valid': False,
                    'stats': area_stats,
                    'reason': f"Insufficient average concealment: {area_stats['avg_concealment']:.2f}"
                }

            # Check threat exposure
            if area_stats['avg_threat'] > primary_reqs['max_threat']:
                return {
                    'is_valid': False,
                    'stats': area_stats,
                    'reason': f"Too much average threat exposure: {area_stats['avg_threat']:.2f}"
                }

        # Area passed all checks
        return {
            'is_valid': True,
            'stats': area_stats
        }

    def _calculate_area_characteristics(self,
                                        area_cells: List[Tuple[int, int]],
                                        terrain: List[List[TerrainInfo]]) -> Dict:
        """
        Calculate comprehensive characteristics for an area.

        Args:
            area_cells: List of cells in the area
            terrain: Terrain grid

        Returns:
            Dictionary containing:
            - avg_cover: Average cover value
            - avg_concealment: Average concealment value
            - avg_movement_cost: Average movement cost
            - elevation_percentage: Percentage of elevated cells
            - avg_threat: Average threat level
            - quality_score: Overall area quality score
        """
        total_cover = 0
        total_concealment = 0
        total_movement_cost = 0
        total_threat = 0
        elevated_cells = 0

        for x, y in area_cells:
            terrain_info = terrain[y][x]
            total_cover += terrain_info.cover_bonus
            total_concealment += (1 - terrain_info.visibility_factor)
            total_movement_cost += terrain_info.movement_cost
            total_threat += self.threat_matrix[y][x]
            if terrain_info.elevation_type.value > 0:
                elevated_cells += 1

        num_cells = len(area_cells)

        # Calculate averages
        avg_cover = total_cover / num_cells
        avg_concealment = total_concealment / num_cells
        avg_movement_cost = total_movement_cost / num_cells
        elevation_percentage = elevated_cells / num_cells
        avg_threat = total_threat / num_cells

        # Calculate rough quality score
        quality_score = (avg_cover * 0.3 +
                         avg_concealment * 0.3 +
                         (elevation_percentage * 0.2) +
                         (1 - (avg_movement_cost / 3)) * 0.1 +
                         (1 - avg_threat) * 0.1)

        return {
            'avg_cover': avg_cover,
            'avg_concealment': avg_concealment,
            'avg_movement_cost': avg_movement_cost,
            'elevation_percentage': elevation_percentage,
            'avg_threat': avg_threat,
            'quality_score': quality_score
        }

    def _evaluate_position(self, area_cells: List[Tuple[int, int]],
                           terrain: List[List[TerrainInfo]],
                           position_type: TacticalPositionType,
                           objective: Tuple[int, int]) -> PositionEvaluation:
        """
        Evaluate complete characteristics of a position.

        Args:
            area_cells: List of cells in the position area
            terrain: Terrain grid
            position_type: Type of tactical position
            objective: Target position

        Returns:
            Dictionary containing:
            - quality_score: Overall position quality
            - coverage_arc: Tuple of arc angles
            - covered_threats: List of engageable threats
            - engagement_quality: Quality of engagements
            - has_elevation: Whether position has elevation advantage
        """
        # Calculate center position
        center_x = sum(x for x, _ in area_cells) // len(area_cells)
        center_y = sum(y for _, y in area_cells) // len(area_cells)
        center = (center_x, center_y)

        # Get requirements and area characteristics
        requirements = position_type.get_requirements()
        quality_factors = requirements['quality_factors']
        area_stats = self._calculate_area_characteristics(area_cells, terrain)

        # Initialize position score components
        score_components: Dict[str, float] = {}

        if requirements['primary']['type'] == 'engagement':
            # Evaluate engagement capability
            engagement = self._evaluate_engagement(center, objective, terrain)
            los_result = self._check_line_of_sight_quality(center, objective, terrain)

            score_components.update({
                'los_quality': float(los_result['los_quality']),
                'elevation_advantage': 1.0 if area_stats['elevation_percentage'] > 0 else 0.0,
                'cover': float(area_stats['avg_cover']),
                'concealment': float(area_stats['avg_concealment'])
            })

            # Calculate weighted score
            quality_score = 0.0
            for factor, value in score_components.items():
                if factor in quality_factors:
                    factor_config = quality_factors[factor]
                    min_val = factor_config.get('min')
                    if min_val is None or value >= min_val:
                        quality_score += value * factor_config['weight']

            # Get engagement arc
            arc_result = self._calculate_engagement_arc(center, objective, terrain)
            arc = arc_result['arc']
            # Ensure we have a proper tuple of two floats
            coverage_arc: Tuple[float, float] = (float(arc[0]), float(arc[1]))

            return {
                'quality_score': float(quality_score),
                'coverage_arc': coverage_arc,
                'covered_threats': [objective] if engagement['can_engage'] else [],
                'engagement_quality': float(engagement['quality_score']),
                'has_elevation': bool(area_stats['elevation_percentage'] > 0),
                'distance_to_objective': float(engagement['range'])
            }

        else:  # ASSAULT
            score_components = {
                'cover': float(area_stats['avg_cover']),
                'concealment': float(area_stats['avg_concealment']),
                'threat_exposure': float(1.0 - area_stats['avg_threat'])
            }

            # Calculate weighted score
            quality_score = 0.0
            for factor, value in score_components.items():
                if factor in quality_factors:
                    factor_config = quality_factors[factor]
                    max_val = factor_config.get('max')
                    min_val = factor_config.get('min')

                    if ((max_val is None or value <= max_val) and
                            (min_val is None or value >= min_val)):
                        quality_score += value * factor_config['weight']

            # For assault positions, use a default arc
            coverage_arc: Tuple[float, float] = (0.0, 0.0)

            return {
                'quality_score': float(quality_score),
                'coverage_arc': coverage_arc,
                'covered_threats': [],
                'engagement_quality': 0.0,
                'has_elevation': bool(area_stats['elevation_percentage'] > 0),
                'distance_to_objective': float(math.sqrt(
                    (center_x - objective[0]) ** 2 +
                    (center_y - objective[1]) ** 2
                ) * 10)
            }

    def _has_enemy_engagement(self, x: int, y: int,
                              terrain: List[List[TerrainInfo]]) -> bool:
        """
        Quick initial check if position can engage any enemy.
        Used for fast filtering of positions during initial search.

        Args:
            x, y: Position coordinates to check
            terrain: Terrain grid

        Returns:
            Boolean indicating if position has any possible engagement
        """
        for threat in self.enemy_threats:
            # Calculate distance to enemy
            dx = threat.position[0] - x
            dy = threat.position[1] - y
            dist = math.sqrt(dx * dx + dy * dy)

            # Quick range check using weapon ranges
            if dist <= 150:  # Extended from 80 to allow for longer range weapons
                # Enhanced line of sight check
                los_result = self._check_line_of_sight_quality(
                    (x, y), threat.position, terrain)

                if los_result['has_los'] and los_result['los_quality'] >= 0.2:  # Reduced threshold
                    return True

        return False

    def _evaluate_engagement(self, position: Tuple[int, int],
                             target: Tuple[int, int],
                             terrain: List[List[TerrainInfo]]) -> Dict:
        """Evaluate ability to engage a specific target."""
        x, y = position

        # Calculate distance
        dx = target[0] - x
        dy = target[1] - y
        dist = math.sqrt(dx * dx + dy * dy) * 10  # Convert to meters

        # Check line of sight
        los_result = self._check_line_of_sight_quality(position, target, terrain)

        if not los_result['has_los']:
            return {
                'can_engage': False,
                'quality_score': 0.0,
                'range': dist,
                'los_quality': los_result['los_quality'],
                'elevation_advantage': 0
            }

        # Check terrain at position and target
        position_terrain = terrain[y][x]
        target_terrain = terrain[target[1]][target[0]]
        elevation_advantage = (
                position_terrain.elevation_type.value -
                target_terrain.elevation_type.value
        )

        # Calculate range score
        if dist <= 300:  # Close range
            range_score = 1.0
        elif dist <= 500:  # Medium range
            range_score = 0.8
        elif dist <= 800:  # Long range
            range_score = 0.6
        elif dist <= 2000:  # Extended range
            range_score = 0.4
        else:  # Beyond effective range
            return {
                'can_engage': False,
                'quality_score': 0.0,
                'range': dist,
                'los_quality': los_result['los_quality'],
                'elevation_advantage': elevation_advantage
            }

        # Calculate final quality score
        quality_score = (
                range_score * 0.4 +
                los_result['los_quality'] * 0.3 +
                min(1.0, max(0.0, 0.5 + elevation_advantage * 0.25)) * 0.3
        )

        return {
            'can_engage': True,
            'quality_score': quality_score,
            'range': dist,
            'los_quality': los_result['los_quality'],
            'elevation_advantage': elevation_advantage
        }

    def _evaluate_engagement_capabilities(self,
                                          center: Tuple[int, int],
                                          area_cells: List[Tuple[int, int]],
                                          terrain: List[List[TerrainInfo]]) -> Dict:
        """
        Evaluate how well position can engage enemy positions.

        Args:
            center: Center position of area
            area_cells: All cells in the area
            terrain: Terrain grid

        Returns:
            Dictionary containing:
            - can_engage: Whether any threats can be engaged
            - num_threats: Number of engageable threats
            - quality_score: Overall engagement quality (0-1)
            - best_engagements: List of the best engagement opportunities
            - covered_arcs: List of coverage arcs
        """
        engageable_threats = []
        best_engagements = []
        quality_scores = []

        for threat in self.enemy_threats:
            # Find the best engagement position within area
            best_engagement = {'quality_score': 0.0}
            best_position = None

            for x, y in area_cells:
                engagement = self._can_engage_threat(
                    (x, y), threat.position, terrain)

                if engagement['can_engage']:
                    if engagement['quality_score'] > best_engagement['quality_score']:
                        best_engagement = engagement
                        best_position = (x, y)

            if best_position:
                engageable_threats.append(threat)
                quality_scores.append(best_engagement['quality_score'])
                best_engagements.append({
                    'threat': threat,
                    'position': best_position,
                    'engagement': best_engagement
                })

        # Calculate coverage arcs
        coverage_arcs = []
        for eng in best_engagements:
            arc = self._calculate_engagement_arc(
                eng['position'],
                eng['threat'].position,
                terrain
            )
            coverage_arcs.append(arc)

        return {
            'can_engage': len(engageable_threats) > 0,
            'num_threats': len(engageable_threats),
            'quality_score': max(quality_scores) if quality_scores else 0.0,
            'best_engagements': best_engagements,
            'covered_arcs': coverage_arcs
        }

    def _can_engage_threat(self, position: Tuple[int, int],
                           threat_pos: Tuple[int, int],
                           terrain: List[List[TerrainInfo]]) -> Dict:
        """
        Evaluate ability to engage specific threat with detailed analysis.

        Args:
            position: Position to check
            threat_pos: Enemy threat position
            terrain: Terrain grid

        Returns:
            Dictionary containing:
            - can_engage: Whether threat can be engaged
            - quality_score: Quality of engagement (0-1)
            - range: Distance to threat
            - range_quality: Quality of range (0-1)
            - has_elevation: Elevation advantage exists
            - elevation_score: Quality of elevation advantage
            - los_quality: Quality of line of sight
            - engagement_arc: Width of engagement arc
        """
        x, y = position
        terrain_info = terrain[y][x]

        # Calculate distance
        dist = math.sqrt(
            (position[0] - threat_pos[0]) ** 2 +
            (position[1] - threat_pos[1]) ** 2
        )

        # Check range bands
        if dist > 200:  # Maximum engagement range
            return {
                'can_engage': False,
                'quality_score': 0.0,
                'reason': 'Out of range',
                'range': dist
            }

        # Optimal range bands (in spaces/meters)
        optimal_range = {
            'close': (50, 100),  # 500-1000m
            'medium': (100, 150),  # 1000-1500m
            'long': (150, 200)  # 1500-2000m
        }

        # Calculate range quality
        if optimal_range['close'][0] <= dist <= optimal_range['close'][1]:
            range_quality = 1.0
        elif optimal_range['medium'][0] <= dist <= optimal_range['medium'][1]:
            range_quality = 0.8
        else:
            range_quality = 0.6

        # Check line of sight quality
        los_result = self._check_line_of_sight_quality(position, threat_pos, terrain)
        if not los_result['has_los']:
            return {
                'can_engage': False,
                'quality_score': 0.0,
                'reason': 'No line of sight',
                'los_quality': 0.0,
                'range': dist
            }

        # Calculate elevation advantage
        elevation_advantage = terrain_info.elevation_type.value - \
                              terrain[threat_pos[1]][threat_pos[0]].elevation_type.value
        elevation_score = min(1.0, max(0.0, 0.5 + elevation_advantage * 0.25))

        # Calculate engagement arc
        arc_result = self._calculate_engagement_arc(position, threat_pos, terrain)

        # Calculate final quality score
        quality_score = (
                range_quality * 0.3 +
                los_result['los_quality'] * 0.3 +
                elevation_score * 0.2 +
                arc_result['arc_quality'] * 0.2
        )

        return {
            'can_engage': True,
            'quality_score': quality_score,
            'range': dist,
            'range_quality': range_quality,
            'has_elevation': elevation_advantage > 0,
            'elevation_score': elevation_score,
            'los_quality': los_result['los_quality'],
            'engagement_arc': arc_result['arc_width'],
            'arc_quality': arc_result['arc_quality']
        }

    def _check_line_of_sight(self, start: Tuple[int, int],
                             end: Tuple[int, int],
                             terrain: List[List[TerrainInfo]]) -> bool:
        """
        Basic line of sight check using Bresenham's algorithm.

        Args:
            start: Starting position (x, y)
            end: Ending position (x, y)
            terrain: Terrain grid

        Returns:
            Boolean indicating if line of sight exists
        """
        x1, y1 = start
        x2, y2 = end

        if not (0 <= x2 < self.env_width and 0 <= y2 < self.env_height):
            return False

        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        for _ in range(n):
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

            if not (0 <= x < self.env_width and 0 <= y < self.env_height):
                return False

            terrain_info = terrain[y][x]
            if terrain_info.terrain_type in [TerrainType.WOODS, TerrainType.STRUCTURE]:
                return False

        return True

    def _check_line_of_sight_quality(self, position: Tuple[int, int],
                                     target: Tuple[int, int],
                                     terrain: List[List[TerrainInfo]]) -> Dict:
        """
        Enhanced line of sight check that considers quality of sight line.

        Args:
            position: Starting position
            target: Target position
            terrain: Terrain grid

        Returns:
            Dictionary containing:
            - has_los: Basic line of sight exists
            - los_quality: Quality of line of sight (0-1)
            - interference: List of interfering terrain features
        """
        x1, y1 = position
        x2, y2 = target

        if not (0 <= x2 < self.env_width and 0 <= y2 < self.env_height):
            return {
                'has_los': False,
                'los_quality': 0.0,
                'interference': ['Out of bounds']
            }

        # Use Bresenham's algorithm to check line of sight
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)
        x, y = x1, y1
        n = 1 + dx + dy
        x_inc = 1 if x2 > x1 else -1
        y_inc = 1 if y2 > y1 else -1
        error = dx - dy
        dx *= 2
        dy *= 2

        # Track line of sight quality
        total_points = 0
        blocked_points = 0
        interference = []

        for _ in range(n):
            total_points += 1

            # Check if point is in bounds
            if not (0 <= x < self.env_width and 0 <= y < self.env_height):
                return {
                    'has_los': False,
                    'los_quality': 0.0,
                    'interference': ['Path out of bounds']
                }

            # Check terrain at current point
            terrain_info = terrain[y][x]

            # Complete blockers
            if terrain_info.terrain_type in [TerrainType.STRUCTURE]:
                return {
                    'has_los': False,
                    'los_quality': 0.0,
                    'interference': [f'Blocked by {terrain_info.terrain_type.name}']
                }

            # Partial blockers reduce quality
            if terrain_info.terrain_type in [TerrainType.WOODS, TerrainType.DENSE_VEG]:
                blocked_points += 1
                interference.append(f'Interference from {terrain_info.terrain_type.name}')

            # Move to next point
            if error > 0:
                x += x_inc
                error -= dy
            else:
                y += y_inc
                error += dx

        # Calculate final line of sight quality
        if blocked_points >= total_points * 0.8:  # More than 80% blocked
            return {
                'has_los': False,
                'los_quality': 0.0,
                'interference': interference
            }

        los_quality = 1.0 - (blocked_points / total_points)
        return {
            'has_los': True,
            'los_quality': los_quality,
            'interference': interference
        }

    def _calculate_engagement_arc(self, position: Tuple[int, int],
                                  target: Tuple[int, int],
                                  terrain: List[List[TerrainInfo]]) -> Dict[str, Union[Tuple[float, float], float]]:
        """
        Calculate the engagement arc available for the position.

        Args:
            position: Position being evaluated
            target: Target position
            terrain: Terrain grid

        Returns:
            Dictionary containing:
            - arc_width: Width of engagement arc in degrees
            - arc_quality: Quality of arc (0-1)
            - key_angles: List of significant angles in arc
        """
        x, y = position
        target_x, target_y = target

        # Calculate angle to target
        target_angle = math.degrees(math.atan2(
            target_y - y,
            target_x - x
        )) % 360

        # Calculate standard military arc (30 degrees each side of target)
        arc_width = 60.0
        start_angle = (target_angle - arc_width / 2) % 360
        end_angle = (target_angle + arc_width / 2) % 360

        # For visualization clarity, ensure arc crosses the target
        if start_angle > end_angle:
            # If arc crosses 0/360, adjust to ensure proper rendering
            if abs(target_angle - start_angle) < abs(target_angle - end_angle):
                end_angle += 360
            else:
                start_angle -= 360

        return {
            'arc': (float(start_angle), float(end_angle)),
            'arc_width': float(arc_width),
            'arc_quality': 1.0,
            'target_angle': float(target_angle)
        }

    def _find_best_engagement_arc(self,
                                  position: Tuple[int, int],
                                  target_angle: float,
                                  terrain: List[List[TerrainInfo]],
                                  target: Tuple[int, int]) -> Optional[Tuple[float, float]]:
        """
        Find the best engagement arc centered near the target angle.

        Args:
            position: Position to check from
            target_angle: Base angle to target
            terrain: Terrain grid
            target: Target position to engage

        Returns:
            Tuple of (start_angle, end_angle) in degrees, or None if no valid arc found
        """
        # Standard arc width for military operations
        standard_width = 60.0

        # Try standard arc first
        base_arc = (
            (target_angle - standard_width / 2) % 360,
            (target_angle + standard_width / 2) % 360
        )

        if self._validate_engagement_arc(position, base_arc, terrain, target):
            return base_arc

        # If standard arc doesn't work, try different widths
        for width in [45.0, 30.0, 90.0]:  # Try narrower and wider arcs
            test_arc = (
                (target_angle - width / 2) % 360,
                (target_angle + width / 2) % 360
            )
            if self._validate_engagement_arc(position, test_arc, terrain, target):
                return test_arc

        # If centered arcs don't work, try offset arcs
        for offset in [-15.0, 15.0, -30.0, 30.0]:  # Try different offsets
            test_arc = (
                (target_angle - standard_width / 2 + offset) % 360,
                (target_angle + standard_width / 2 + offset) % 360
            )
            if self._validate_engagement_arc(position, test_arc, terrain, target):
                return test_arc

        # Try to find any valid arc through systematic search
        check_points = []
        x, y = position
        range_to_target = math.sqrt(
            (target[0] - x) ** 2 +
            (target[1] - y) ** 2
        )

        # Check points at regular intervals
        for angle in range(0, 360, 5):
            check_x = x + range_to_target * math.cos(math.radians(angle))
            check_y = y + range_to_target * math.sin(math.radians(angle))

            if (0 <= check_x < self.env_width and
                    0 <= check_y < self.env_height and
                    self._check_line_of_sight(position, (int(check_x), int(check_y)), terrain)):
                check_points.append(angle)

        # Find continuous arcs from check points
        if check_points:
            # Find the longest continuous arc that includes the target angle
            best_arc = None
            best_width = 0

            start_idx = 0
            while start_idx < len(check_points):
                end_idx = start_idx
                while (end_idx < len(check_points) - 1 and
                       check_points[end_idx + 1] - check_points[end_idx] <= 10):  # Allow small gaps
                    end_idx += 1

                arc_start = check_points[start_idx]
                arc_end = check_points[end_idx]
                arc_width = (arc_end - arc_start) % 360

                # Check if this arc can engage the target
                if self._angle_in_arc(target_angle, (arc_start, arc_end)):
                    if arc_width > best_width:
                        best_arc = (arc_start, arc_end)
                        best_width = arc_width

                start_idx = end_idx + 1

            if best_arc is not None:
                return best_arc

        return None

    def _angle_in_arc(self, angle: float, arc: Tuple[float, float]) -> bool:
        """Check if an angle falls within an arc."""
        start_angle, end_angle = arc

        if start_angle <= end_angle:
            return start_angle <= angle <= end_angle
        else:  # Arc crosses 0/360
            return angle >= start_angle or angle <= end_angle

    def _calculate_target_angle(self, position: Tuple[int, int], target: Tuple[int, int]) -> float:
        """Calculate angle to target in degrees."""
        x, y = position
        target_x, target_y = target
        return math.degrees(math.atan2(
            target_y - y,
            target_x - x
        )) % 360

    def _validate_engagement_arc(self, position: Tuple[int, int],
                                 arc: Tuple[float, float],
                                 terrain: List[List[TerrainInfo]],
                                 target: Tuple[int, int]) -> bool:
        """Validate that an engagement arc can effectively engage the target."""
        x, y = position
        start_angle, end_angle = arc

        # Check if target is within arc
        target_angle = math.degrees(math.atan2(
            target[1] - y,
            target[0] - x
        )) % 360

        # Handle arc that crosses 0/360
        if start_angle > end_angle:
            in_arc = target_angle >= start_angle or target_angle <= end_angle
        else:
            in_arc = start_angle <= target_angle <= end_angle

        if not in_arc:
            return False

        # Check if we can actually engage along the arc
        return self._check_line_of_sight((x, y), target, terrain)

    def visualize_analysis(self,
                           terrain: List[List[TerrainInfo]],
                           positions: List[TacticalPosition],
                           save_path: str = 'tactical_analysis.png'):
        """
        Create enhanced visualization of tactical analysis with improved layout.
        - Main plot takes up most of the figure
        - Colorbar (1/3 width) beneath main plot, left-aligned
        - Legend beneath main plot, right-aligned
        """
        # Create figure with larger size
        fig = plt.figure(figsize=(20, 16))  # Increased overall figure size

        # Create grid layout with better proportions
        # Main plot: 16 rows, full width
        # Bottom row: Split between colorbar (left) and legend (right)
        gs = plt.GridSpec(20, 24)  # 20 rows, 24 columns grid

        # Create axes:
        # Main plot uses most of the space
        ax = fig.add_subplot(gs[0:18, 0:24])
        # Colorbar takes left third of bottom space
        cax = fig.add_subplot(gs[18:20, 0:8])
        # Legend takes right third of bottom space
        legend_ax = fig.add_subplot(gs[18:20, 16:24])
        legend_ax.axis('off')  # Hide legend axis

        # Plot base terrain
        terrain_img = self._create_terrain_image(terrain)
        main_img = ax.imshow(terrain_img, extent=(0, self.env_width, 0, self.env_height))

        # Plot normalized threat overlay
        normalized_threat = self._normalize_threat_matrix()
        threat_cmap = plt.colormaps['Reds']
        threat_overlay = ax.imshow(
            normalized_threat,
            alpha=0.3,
            cmap=threat_cmap,
            extent=(0, self.env_width, 0, self.env_height)
        )

        # Add horizontal colorbar, left-aligned
        plt.colorbar(threat_overlay, cax=cax, orientation='horizontal', label='Threat Level')

        # Plot enemy positions
        for threat in self.enemy_threats:
            self._plot_enemy_position(ax, threat)

        # Plot objective
        ax.plot(self.objective[0], self.objective[1], 'r*', markersize=15, label='Objective')

        # Sort positions by quality score and take top 3
        top_positions = sorted(positions, key=lambda p: p.quality_score, reverse=True)[:3]

        # Position type colors
        position_colors = {
            TacticalPositionType.FIRE_SUPPORT: 'blue',
            TacticalPositionType.SUPPORT_BY_FIRE: 'green',
            TacticalPositionType.ASSAULT: 'yellow'
        }

        # Plot top 5 positions
        for i, pos in enumerate(top_positions, 1):
            color = position_colors[pos.position_type]
            x, y = pos.position

            # Plot position center with ranking
            ax.plot(x, y, 'o', color=color, markersize=10,
                    label=f'#{i} {pos.position_type.value}\n(Score: {pos.quality_score:.2f})')

            # Add ranking number
            ax.text(x + 1, y + 1, str(i), color=color, fontweight='bold', fontsize=12)

            # Plot area outline
            area_size = 7 if pos.unit_size == UnitSize.TEAM else 12
            rect = Rectangle(
                (x - area_size // 2, y - area_size // 2),
                area_size,
                area_size,
                fill=False,
                color=color,
                linewidth=2
            )
            ax.add_patch(rect)

            # Plot coverage arc if applicable
            if pos.position_type in [TacticalPositionType.FIRE_SUPPORT, TacticalPositionType.SUPPORT_BY_FIRE]:
                self._plot_coverage_arc(ax, pos, color)

        # Customize main plot
        ax.set_title('Tactical Position Analysis', pad=20, fontsize=14)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Get handles and labels from main plot
        handles, labels = ax.get_legend_handles_labels()

        # Create legend in the right-aligned legend axis
        legend = legend_ax.legend(handles, labels,
                                  loc='center',
                                  bbox_to_anchor=(0.5, 0.5),
                                  fontsize=10,
                                  title='Legend',
                                  title_fontsize=12,
                                  ncol=2)  # Use 2 columns to make legend more compact

        # Adjust layout
        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.15, top=0.9)

        # Save figure with high resolution
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

    def _normalize_threat_matrix(self) -> np.ndarray:
        """Normalize threat matrix to range [0,1]."""
        if self.threat_matrix.max() > 0:
            return self.threat_matrix / self.threat_matrix.max()
        return self.threat_matrix

    def _plot_enemy_position(self, ax: plt.Axes, threat: EnemyThreat) -> None:
        """
        Plot enemy position with observation and engagement ranges.

        Args:
            ax: Matplotlib axes object to plot on
            threat: Enemy threat object to plot
        """
        x, y = threat.position

        # Plot position with larger, more visible marker
        ax.plot(x, y, 'r^', markersize=12, markeredgecolor='black',
                label=f'Enemy Position (Obs: {threat.observation_range * 10}m, Eng: {threat.engagement_range * 10}m)')

        # Plot observation range with dashed line
        obs_circle = Circle(
            (x, y),
            threat.observation_range,
            fill=False,
            linestyle='--',
            color='red',
            alpha=0.5,
            linewidth=1.5
        )
        ax.add_patch(obs_circle)

        # Plot engagement range with solid line
        eng_circle = Circle(
            (x, y),
            threat.engagement_range,
            fill=False,
            linestyle='-',
            color='red',
            alpha=0.7,
            linewidth=2
        )
        ax.add_patch(eng_circle)

    def _plot_coverage_arc(self, ax: plt.Axes, position: TacticalPosition, color: str):
        """
        Plot coverage arc for a tactical position, respecting map boundaries and weapon ranges.

        Args:
            ax: Matplotlib axes to plot on
            position: TacticalPosition object
            color: Color to use for the arc
        """
        x, y = position.position
        start_angle, end_angle = position.coverage_arc

        # Calculate actual weapon range in cells (convert from meters)
        weapon_range = min(position.max_range / 10, 70)  # Convert to cells, limit to 700m

        # Calculate points along the arc that are within map bounds and range
        theta = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
        arc_points = []

        for angle in theta:
            # Calculate point on arc
            px = x + weapon_range * np.cos(angle)
            py = y + weapon_range * np.sin(angle)

            # Check if point is within map bounds
            if (0 <= px < self.env_width and 0 <= py < self.env_height):
                # Check if point is within weapon range of position
                dist = np.sqrt((px - x) ** 2 + (py - y) ** 2)
                if dist <= weapon_range:
                    arc_points.append((px, py))

        if arc_points:
            # Draw arc
            points = np.array(arc_points)
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.5, linewidth=2)

            # Draw range lines from center to arc ends
            if len(arc_points) >= 2:
                ax.plot([x, arc_points[0][0]], [y, arc_points[0][1]],
                        color=color, linestyle='--', alpha=0.3)
                ax.plot([x, arc_points[-1][0]], [y, arc_points[-1][1]],
                        color=color, linestyle='--', alpha=0.3)

                # Draw line to objective to show orientation
                ax.plot([x, self.objective[0]], [y, self.objective[1]],
                        color=color, linestyle=':', alpha=0.5)

    def _plot_bounded_arc(self, ax: plt.Axes, center: Tuple[int, int], radius: float,
                          start_angle: float, end_angle: float, color: str):
        """
        Plot arc segments that are within map boundaries.

        Args:
            ax: Matplotlib axes to plot on
            center: Center point of arc (x, y)
            radius: Radius of arc in cells
            start_angle: Start angle in degrees
            end_angle: End angle in degrees
            color: Color of arc
        """
        x, y = center

        # Create points along the arc
        theta = np.linspace(np.radians(start_angle), np.radians(end_angle), 100)
        arc_x = x + radius * np.cos(theta)
        arc_y = y + radius * np.sin(theta)

        # Filter points to only those within map bounds
        valid_points = []
        for px, py in zip(arc_x, arc_y):
            if (0 <= px < self.env_width and 0 <= py < self.env_height):
                valid_points.append((px, py))

        if valid_points:
            # Draw lines between valid points
            points = np.array(valid_points)
            ax.plot(points[:, 0], points[:, 1], color=color, alpha=0.5, linewidth=2)

            # Draw range lines from center to arc ends
            for p in [valid_points[0], valid_points[-1]]:
                ax.plot([x, p[0]], [y, p[1]], color=color, linestyle='--', alpha=0.3)

    def _create_terrain_image(self, terrain: List[List[TerrainInfo]]) -> np.ndarray:
        """Create RGB image representation of terrain with enhanced colors."""
        terrain_img = np.zeros((self.env_height, self.env_width, 3))

        # Enhanced color mapping for better visibility
        color_map = {
            TerrainType.BARE: [0.9, 0.9, 0.9],  # Light gray
            TerrainType.SPARSE_VEG: [0.8, 0.9, 0.8],  # Light green
            TerrainType.DENSE_VEG: [0.4, 0.7, 0.4],  # Medium green
            TerrainType.WOODS: [0.2, 0.5, 0.2],  # Dark green
            TerrainType.STRUCTURE: [0.6, 0.6, 0.7]  # Blue-gray
        }

        for y in range(self.env_height):
            for x in range(len(terrain[0])):
                terrain_info = terrain[y][x]
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

    def _track_failure(self, position: Tuple[int, int], score: float,
                       reason: str, averages: Dict[str, float]) -> None:
        """Enhanced failure tracking with better scoring."""
        attempt = PositionAttempt(
            position=position,
            score=score,
            failure_reason=reason,
            averages=averages
        )

        # Always add to failures list
        self.top_failures.append(attempt)

        # Sort by score and keep top 10
        self.top_failures.sort(key=lambda x: (-x.score,  # Sort by score descending
                                              x.averages.get('distance', float('inf'))))  # Then by distance

        # Keep only top 10 unique positions
        seen_positions = set()
        unique_failures = []
        for failure in self.top_failures:
            if failure.position not in seen_positions:
                seen_positions.add(failure.position)
                unique_failures.append(failure)
                if len(unique_failures) >= 10:
                    break

        self.top_failures = unique_failures

    def _print_failure_details(self, attempt: PositionAttempt, index: int):
        """
        Print detailed information about a failed position attempt.

        Args:
            attempt: PositionAttempt object containing failure details
            index: Index number for display
        """
        print(f"\n{index}. Position {attempt.position} (Score: {attempt.score:.3f})")

        # Show engagement analysis if available
        if attempt.engagement_stats:
            print(f"   Engagement Analysis:")
            print(f"   - Distance to objective: {attempt.engagement_stats.get('distance', 0):.0f}m")
            print(f"   - Line of sight: {'Yes' if attempt.engagement_stats.get('has_los', False) else 'No'}")
            if 'los_quality' in attempt.engagement_stats:
                print(f"   - Line of sight quality: {attempt.engagement_stats['los_quality']:.2f}")
            print(
                f"   - Can effectively engage: {'Yes' if attempt.engagement_stats.get('can_engage', False) else 'No'}")
            if 'threats_covered' in attempt.engagement_stats:
                print(f"   - Threats covered: {attempt.engagement_stats['threats_covered']}")

        # Show area characteristics
        print(f"   Area Analysis:")
        for key, value in attempt.averages.items():
            if isinstance(value, float):
                print(f"      {key}: {value:.2f}")
            else:
                print(f"      {key}: {value}")

        print(f"   Failure: {attempt.failure_reason}")


# Test Code
def load_test_map(project_root: str, map_filename: str) -> List[List[TerrainInfo]]:
    """
    Load map from CSV file into TerrainInfo grid.

    Args:
        project_root: Root directory of project
        map_filename: Name of CSV map file

    Returns:
        2D list of TerrainInfo objects representing the map
    """
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
                # Convert to properly typed dictionary
                row: Mapping[str, str] = dict(csv_row)

                try:
                    # Access dictionary values
                    x = int(row['x'])
                    y = int(row['y'])
                    terrain_type_str = str(row['terrain_type'])
                    elevation_type_str = str(row['elevation_type'])

                    # Handle new rows
                    if y > current_y:
                        if current_row:
                            terrain_grid.append(current_row)
                        current_row = []
                        current_y = y

                    # Convert strings to enum values
                    terrain_type = TerrainType[terrain_type_str]
                    elevation_type = ElevationType[elevation_type_str]

                    # Create TerrainInfo object
                    terrain_info = TerrainInfo(
                        terrain_type=terrain_type,
                        elevation_type=elevation_type,
                        movement_cost=PathCosts.TERRAIN_MOVEMENT_COSTS[terrain_type],
                        visibility_factor=PathCosts.TERRAIN_VISIBILITY[terrain_type],
                        cover_bonus=PathCosts.TERRAIN_COVER[terrain_type]
                    )

                    current_row.append(terrain_info)

                except (KeyError, ValueError) as e:
                    print(f"Error processing row: {row}")
                    print(f"Error details: {str(e)}")
                    continue

            # Add the last row
            if current_row:
                terrain_grid.append(current_row)

            print(f"Loaded {len(terrain_grid)} rows")
            if terrain_grid:
                print(f"First row has {len(terrain_grid[0])} cells")

    except Exception as e:
        print(f"Error loading map: {str(e)}")
        import traceback
        traceback.print_exc()
        return []

    return terrain_grid


def test_single_position():
    """Test single position evaluation with more detailed output."""
    print("\nTesting single tactical position analyzer with unit size and enemy threats...")

    # Load terrain
    map_filename = 'generated_map.csv'
    print(f"Loading terrain from: {map_filename}")
    terrain = load_test_map(project_root, map_filename)

    if not terrain:
        print("Failed to load terrain map!")
        return

    env_height = len(terrain)
    env_width = len(terrain[0])
    print(f"Loaded terrain map: {env_width}x{env_height} cells")

    # Create analyzer instance with objective
    objective = (353, 94)
    analyzer = TacticalPositionAnalyzer(None, env_width, env_height, objective)

    # Initialize threat matrix
    print("\nInitializing threat matrix...")
    enemy_pos = objective
    threat_info = {
        'position': enemy_pos,
        'observation_range': 48,
        'engagement_range': 30
    }

    x, y = enemy_pos
    for i in range(env_height):
        for j in range(env_width):
            distance = math.sqrt((j - x) ** 2 + (i - y) ** 2)
            if distance <= threat_info['engagement_range']:
                analyzer.threat_matrix[i][j] += 0.8 * (1 - distance / threat_info['engagement_range'])
            elif distance <= threat_info['observation_range']:
                analyzer.threat_matrix[i][j] += 0.4 * (1 - distance / threat_info['observation_range'])

    print("Threat matrix initialized")

    # Test position
    test_x, test_y = 384, 36
    print(f"\nTesting position ({test_x}, {test_y}):")

    # Get terrain info
    terrain_info = terrain[test_y][test_x]
    print(f"Terrain type: {terrain_info.terrain_type}")
    print(f"Cover bonus: {terrain_info.cover_bonus}")
    print(f"Elevation: {terrain_info.elevation_type}")

    # Calculate distance to objective
    dist = math.sqrt((test_x - objective[0]) ** 2 + (test_y - objective[1]) ** 2) * 10
    print(f"Distance to objective: {dist:.0f}m")

    # Check line of sight
    los = analyzer._check_line_of_sight_quality((test_x, test_y), objective, terrain)
    print("\nLine of sight analysis:")
    print(f"Has LOS: {los['has_los']}")
    print(f"LOS Quality: {los['los_quality']:.2f}")
    if los['interference']:
        print("Interference factors:")
        for factor in los['interference']:
            print(f"- {factor}")

    # Check engagement capability
    engagement = analyzer._evaluate_engagement((test_x, test_y), objective, terrain)
    print("\nEngagement analysis:")
    print(f"Can engage: {engagement['can_engage']}")
    print(f"Engagement quality: {engagement['quality_score']:.2f}")
    print(f"Range: {engagement['range']:.0f}m")
    print(f"Elevation advantage: {engagement['elevation_advantage']}")

    # Check position suitability
    is_suitable = analyzer._is_suitable_position(
        test_x, test_y,
        terrain,
        TacticalPositionType.FIRE_SUPPORT
    )
    print(f"\nPosition suitable: {is_suitable}")

    if not is_suitable and analyzer.top_failures:
        print("\nFailure details:")
        analyzer._print_failure_details(analyzer.top_failures[0], 1)


def test_tactical_position_analyzer():
    """Test tactical position analysis with unit sizes and enemy threats."""
    print("\nTesting tactical position analyzer with unit sizes and enemy threats...")

    # Load test map
    map_filename = 'generated_map.csv'
    print(f"Loading terrain from: {map_filename}")
    terrain = load_test_map(project_root, map_filename)

    if not terrain:
        print("Failed to load terrain map!")
        return

    # Get terrain dimensions
    env_height = len(terrain)
    env_width = len(terrain[0])
    print(f"Loaded terrain map: {env_width}x{env_height} cells")

    # Create analyzer instance with objective position as tuple
    objective = (342, 51)  # Define objective as a tuple
    analyzer = TacticalPositionAnalyzer(terrain_analyzer=None,
                                        env_width=env_width,
                                        env_height=env_height,
                                        objective=objective)

    # Define enemy positions
    enemy_positions = [
        (346, 50),  # Main position by building
        (340, 48),  # Supporting position
        (342, 52)  # Security position
    ]

    # Create enemy threats
    print("\nCreating enemy threats...")
    for pos in enemy_positions:
        # Create a basic enemy squad
        enemy_squad = Squad(
            squad_id=f"Enemy Squad {len(analyzer.enemy_threats) + 1}",
            leader=Soldier("Enemy Leader", 100, 100, AK12, None, 48, 30, pos, True),
            mg_team=SpecialTeam("MG Team"),
            gl_team=SpecialTeam("GL Team"),
            assault_team=Team("Assault Team",
                              TeamMember(Soldier("Team Leader", 100, 100, AK12, None, 80, 30, pos, True), "Assault"))
        )

        # Create and add EnemyThreat instance
        threat = EnemyThreat(
            position=pos,
            unit=enemy_squad,
            observation_range=48,  # 480m observation range
            engagement_range=30,  # 300m engagement range
            suspected_accuracy=0.8
        )
        analyzer.enemy_threats.append(threat)
        print(f"Added enemy threat at {pos}")

    # Initialize threat matrix
    print("\nInitializing threat matrix...")
    for threat in analyzer.enemy_threats:
        x, y = threat.position
        for i in range(env_height):
            for j in range(env_width):
                distance = math.sqrt((j - x) ** 2 + (i - y) ** 2)

                # Add threat values with distance falloff
                if distance <= threat.engagement_range:
                    analyzer.threat_matrix[i][j] += threat.suspected_accuracy * (1 - distance / threat.engagement_range)
                elif distance <= threat.observation_range:
                    analyzer.threat_matrix[i][j] += (threat.suspected_accuracy * 0.5) * (
                                1 - distance / threat.observation_range)

    print(f"Initialized threat matrix with {len(analyzer.enemy_threats)} enemy threats")

    # Test different types of positions
    test_scenarios = [
        {
            'position_type': TacticalPositionType.FIRE_SUPPORT,
            'unit_size': UnitSize.TEAM,
            'min_range': 60,
            'max_range': 200,
            'description': "Team-sized Fire Support Position"
        },
        {
            'position_type': TacticalPositionType.SUPPORT_BY_FIRE,
            'unit_size': UnitSize.TEAM,
            'min_range': 40,
            'max_range': 75,
            'description': "Team-sized Support by Fire Position"
        },
        {
            'position_type': TacticalPositionType.ASSAULT,
            'unit_size': UnitSize.TEAM,
            'min_range': 0,
            'max_range': 50,
            'description': "Team-sized Assault Position"
        },
        {
            'position_type': TacticalPositionType.FIRE_SUPPORT,
            'unit_size': UnitSize.SQUAD,
            'min_range': 60,
            'max_range': 200,
            'description': "Squad-sized Fire Support Position"
        },
        {
            'position_type': TacticalPositionType.SUPPORT_BY_FIRE,
            'unit_size': UnitSize.SQUAD,
            'min_range': 40,
            'max_range': 75,
            'description': "Squad-sized Support by Fire Position"
        },
        {
            'position_type': TacticalPositionType.ASSAULT,
            'unit_size': UnitSize.SQUAD,
            'min_range': 0,
            'max_range': 50,
            'description': "Squad-sized Assault Position"
        }
    ]

    # Test each scenario
    for scenario in test_scenarios:
        print(f"\n=== Testing {scenario['description']} ===")

        positions = analyzer.find_tactical_positions(
            terrain=terrain,
            position_type=scenario['position_type'],
            unit_size=scenario['unit_size'],
            objective=objective,
            min_range=scenario['min_range'],
            max_range=scenario['max_range']
        )

        # Create visualization for this scenario
        viz_filename = f"tactical_analysis_{scenario['position_type'].value}_{scenario['unit_size'].value}.png"
        analyzer.visualize_analysis(
            terrain=terrain,
            positions=positions,
            save_path=viz_filename
        )

        print(f"Visualization saved as {viz_filename}")

        # Print position details with distance to objective
        print(f"\nFound {len(positions)} suitable positions")
        for i, pos in enumerate(positions[:3]):
            # Calculate distance to objective
            dist_to_obj = math.sqrt(
                (pos.position[0] - objective[0])**2 +
                (pos.position[1] - objective[1])**2
            ) * 10  # Convert to meters

            print(f"\nPosition {i + 1}:")
            print(f"Center: {pos.position}")
            print(f"Distance to Objective: {dist_to_obj:.0f}m")
            print(f"Quality Score: {pos.quality_score:.2f}")
            print(f"Coverage Arc: {pos.coverage_arc[0]:.1f}° to {pos.coverage_arc[1]:.1f}°")
            print(f"Threats Covered: {len(pos.covered_threats)}")


if __name__ == "__main__":
    try:
        print("Starting Tactical Position Analyzer Tests...")
        test_tactical_position_analyzer()
        print("\nAll tests completed successfully!")
        # print("Starting Single Tactical Position Test...")
        # test_single_position()
        # print("\nTest completed successfully!")
    except Exception as e:
        print(f"\nError during testing: {str(e)}")
        import traceback

        traceback.print_exc()
