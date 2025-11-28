# my_team.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import random
import contest.util as util

from contest.capture_agents import CaptureAgent
from contest.game import Directions
from contest.util import nearest_point


#################
# Team creation #
#################

def create_team(first_index, second_index, is_red,
                first='OffensiveReflexAgent', second='DefensiveReflexAgent', num_training=0):
    """
    This function should return a list of two agents that will form the
    team, initialized using firstIndex and secondIndex as their agent
    index numbers.  isRed is True if the red team is being created, and
    will be False if the blue team is being created.

    As a potentially helpful development aid, this function can take
    additional string-valued keyword arguments ("first" and "second" are
    such arguments in the case of this function), which will come from
    the --red_opts and --blue_opts command-line arguments to capture.py.
    For the nightly contest, however, your team will be created without
    any extra arguments, so you should make sure that the default
    behavior is what you want for the nightly contest.
    """
    return [eval(first)(first_index), eval(second)(second_index)]


##########
# Agents #
##########

class ReflexCaptureAgent(CaptureAgent):
    """
    A base class for reflex agents that choose score-maximizing actions
    """

    def _init_(self, index, time_for_computing=.1):
        super()._init_(index, time_for_computing)
        self.start = None

    def register_initial_state(self, game_state):
        self.start = game_state.get_agent_position(self.index)
        CaptureAgent.register_initial_state(self, game_state)

    def choose_action(self, game_state):
        """
        Picks among the actions with the highest Q(s,a).
        """
        actions = game_state.get_legal_actions(self.index)

        # You can profile your evaluation time by uncommenting these lines
        # start = time.time()
        values = [self.evaluate(game_state, a) for a in actions]
        # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

        max_value = max(values)
        best_actions = [a for a, v in zip(actions, values) if v == max_value]

        food_left = len(self.get_food(game_state).as_list())

        if food_left <= 2:
            best_dist = 9999
            best_action = None
            for action in actions:
                successor = self.get_successor(game_state, action)
                pos2 = successor.get_agent_position(self.index)
                dist = self.get_maze_distance(self.start, pos2)
                if dist < best_dist:
                    best_action = action
                    best_dist = dist
            return best_action

        return random.choice(best_actions)

    def get_successor(self, game_state, action):
        """
        Finds the next successor which is a grid position (location tuple).
        """
        successor = game_state.generate_successor(self.index, action)
        pos = successor.get_agent_state(self.index).get_position()
        if pos != nearest_point(pos):
            # Only half a grid position was covered
            return successor.generate_successor(self.index, action)
        else:
            return successor

    def evaluate(self, game_state, action):
        """
        Computes a linear combination of features and feature weights
        """
        features = self.get_features(game_state, action)
        weights = self.get_weights(game_state, action)
        return features * weights

    def get_features(self, game_state, action):
        """
        Returns a counter of features for the state
        """
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        features['successor_score'] = self.get_score(successor)
        return features

    def get_weights(self, game_state, action):
        """
        Normally, weights do not depend on the game state.  They can be either
        a counter or a dictionary.
        """
        return {'successor_score': 1.0}


class OffensiveReflexAgent(ReflexCaptureAgent):
    """
    Safe Forager:
      - avoids ghosts intelligently
      - prioritizes safe food
      - grabs capsules when ghosts are too close
      - returns home when carrying many pellets
      - avoids tunnels when ghosts are nearby
      - avoids loops / getting stuck
    """

    def __init__(self, index, time_for_computing=.1):
        super().__init__(index, time_for_computing)
        self.prev_positions = []

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)
        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # TRACK POSITION HISTORY TO AVOID LOOPS
        self.prev_positions = (self.prev_positions + [my_pos])[-8:]
        print( "Position History: ", self.prev_positions )
        if self.prev_positions.count(my_pos) >= 3:
            features['looping'] = 1

        # ============================================================
        # 1. FOOD TARGETING
        # ============================================================
        food = self.get_food(successor).as_list()
        features['successor_score'] = -len(food)  # more food eaten → higher score
        if len(food) > 0:
            # pick the SAFE nearest food (ghost proximity-weighted)
            safe_food_dists = []
            for f in food:
                dist = self.get_maze_distance(my_pos, f)
                if dist < 1:
                    continue
                if self._food_is_safe(successor, my_pos, f):
                    safe_food_dists.append(dist)

            if safe_food_dists:
                features['distance_to_food'] = min(safe_food_dists)
            else:
                # fallback: use closest food even if unsafe
                dists = [self.get_maze_distance(my_pos, f) for f in food]
                features['distance_to_food'] = min(dists)

        # ============================================================
        # 2. GHOST AVOIDANCE
        # ============================================================
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghosts = []
        scared_ghosts = []
        for e in enemies:
            if e.get_position() is None:
                continue
            if not e.is_pacman:
                if e.scared_timer > 5:
                    scared_ghosts.append(e)
                else:
                    ghosts.append(e)

        # nearest dangerous ghost
        if ghosts:
            g_dists = [self.get_maze_distance(my_pos, g.get_position()) for g in ghosts]
            features['ghost_dist'] = min(g_dists)
        else:
            features['ghost_dist'] = 999

        # ============================================================
        # 3. CAPSULE LOGIC
        # ============================================================
        capsules = self.get_capsules(successor)
        if capsules:
            cap_dists = [self.get_maze_distance(my_pos, c) for c in capsules]
            features['distance_to_capsule'] = min(cap_dists)
        else:
            features['distance_to_capsule'] = 999

        # ============================================================
        # 4. CARRYING / RETURN HOME WHEN FULL
        # ============================================================
        # Berkeley contest states include num_carrying
        cur = game_state.get_agent_state(self.index)
        carrying = getattr(cur, "num_carrying", 0)
        features['carrying'] = carrying

        # distance back to home border
        features['home_dist'] = self._distance_to_home(successor, my_pos)

        # ============================================================
        # 5. BAD MOVEMENTS
        # ============================================================
        if action == Directions.STOP:
            features['stop'] = 1

        reverse = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == reverse:
            features['reverse'] = 1
    
        width = game_state.data.layout.width
        mid = width // 2
        if self.red:
            features['on_offense'] = my_pos[0] if my_pos[0] < mid else mid
        else:
            features['on_offense'] = mid -my_pos[0] if my_pos[0] > mid else mid
            
        return features


    def get_weights(self, game_state, action):
        cur_state = game_state.get_agent_state(self.index)
        carrying = getattr(cur_state, "num_carrying", 0)

        # Distance to the nearest ghost
        successor = self.get_successor(game_state, action)
        my_pos = successor.get_agent_state(self.index).get_position()
        
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        ghost_dists = []
        scared_ghosts = []
        for e in enemies:
            if e.get_position() is None or e.is_pacman:
                continue
            dist = self.get_maze_distance(my_pos, e.get_position())
            if e.scared_timer > 5:
                scared_ghosts.append(e)
            else:
                ghost_dists.append(dist)
            
        nearest_ghost_dist = min(ghost_dists) if ghost_dists else 999

        # Base weights
        weights = {
            'successor_score': 200,
            'distance_to_food': -5.0,
            'ghost_dist': 7.0,
            'distance_to_capsule': 0.0,
            'carrying': 10.0,
            'home_dist': 0.0,
            'stop': -100000,
            'reverse': -4,
            'looping': -100000,
            'on_offense': 100
        }

        # if carrying more than 2 balls priorize returning home
        if carrying > 2:
            weights['home_dist'] = -10.0
            weights['distance_to_food'] = 0.0  # stop searching food
            weights['ghost_dist'] = 10.0 # avoid ghosts more strongly
            weights['on_offense'] = 0.0  # focus on returning home

        # If ghost very near priorize escape from it
        if nearest_ghost_dist <= 2:
            weights['distance_to_food'] = 0  # stop searching food
            weights['successor_score'] = 0 # score is irrelevant
            weights['ghost_dist'] = 20.0  # strong avoidance
            weights['distance_to_capsule'] = -5.0  # go for capsule
            weights['on_offense'] = 0.0  # focus on escape
            
        if scared_ghosts:
            weights['ghost_dist'] = 0.0  # not worried about scared ghosts

        return weights

    # ============================================================
    # HELPER METHODS
    # ============================================================

    def _distance_to_home(self, game_state, pos):
        width = game_state.data.layout.width
        mid = width // 2
        if self.red:
            bx = mid - 1
        else:
            bx = mid
        ys = []
        for y in range(game_state.data.layout.height):
            if not game_state.has_wall(bx, y):
                ys.append((bx, y))
        if not ys:
            return self.get_maze_distance(pos, self.start)
        return min(self.get_maze_distance(pos, y) for y in ys)

    def _food_is_safe(self, game_state, my_pos, food_pos):
        """
        Food is safe if no ghost can beat me to it.
        """
        enemies = [game_state.get_agent_state(i) for i in self.get_opponents(game_state)]
        my_dist = self.get_maze_distance(my_pos, food_pos)

        for e in enemies:
            if e.get_position() is None:
                continue
            if not e.is_pacman and e.scared_timer == 0:
                gdist = self.get_maze_distance(e.get_position(), food_pos)
                if gdist < my_dist + 2:
                    return False
        return True



class DefensiveReflexAgent(ReflexCaptureAgent):
    """
    A reflex agent that keeps its side Pacman-free. Again,
    this is to give you an idea of what a defensive agent
    could be like.  It is not the best or only way to make
    such an agent.
    """

    def get_features(self, game_state, action):
        features = util.Counter()
        successor = self.get_successor(game_state, action)

        my_state = successor.get_agent_state(self.index)
        my_pos = my_state.get_position()

        # on defense?
        features['on_defense'] = 1
        if my_state.is_pacman:
            features['on_defense'] = 0

        # visible enemies and invaders (enemy pacmen on our side)
        enemies = [successor.get_agent_state(i) for i in self.get_opponents(successor)]
        invaders = [e for e in enemies if e.is_pacman and e.get_position() is not None]
        features['num_invaders'] = len(invaders)

        if len(invaders) > 0:
            dists = [self.get_maze_distance(my_pos, a.get_position()) for a in invaders]
            features['invader_distance'] = min(dists)
        else:
            # Patrol: compute distance to a chosen patrol point on border (midline chokepoint)
            patrol_point = self._choose_patrol_point(game_state)
            features['distance_to_patrol'] = self.get_maze_distance(my_pos, patrol_point)

        # Are we stuck / stopping?
        if action == Directions.STOP:
            features['stop'] = 1
        rev = Directions.REVERSE[game_state.get_agent_state(self.index).configuration.direction]
        if action == rev:
            features['reverse'] = 1

        # small penalty for moving onto a food-eaten location (encourage staying near chokepoints)
        # not always available: check if we can get position on map
        return features

    def get_weights(self, game_state, action):
        #return {'num_invaders': -1000, 'on_defense': 100, 'invader_distance': -10, 'stop': -100, 'reverse': -2}
        # Strong priorities: catch invaders, stay on defense, avoid stop
        return {
            'num_invaders': -1000,
            'on_defense': 100,
            'invader_distance': -10,
            'distance_to_patrol': -1.5,
            'stop': -100,
            'reverse': -2
        }
    def _choose_patrol_point(self, game_state):
        """
        Choose a reasonable patrol point along the midline of our side.
        Strategy: select a non-wall point near the middle of the board on our side.
        """
        width = game_state.data.layout.width
        height = game_state.data.layout.height
        mid = width // 2
        # pick column on our side near the border
        if self.red:
            border_x = mid - 1
        else:
            border_x = mid

        # choose the tallest open stretch of mid column, then pick a middle y
        open_positions = [(border_x, y) for y in range(height) if not game_state.has_wall(border_x, y)]
        if not open_positions:
            # fallback to start
            return self.start
        ys = [p[1] for p in open_positions]
        median_y = sorted(ys)[len(ys)//2]

        return (border_x, median_y)
