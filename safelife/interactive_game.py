import os
import sys
import glob
import textwrap
import time
from types import SimpleNamespace
import numpy as np

from .game_physics import SafeLife
from . import ascii_renderer
from . import rgb_renderer
from .gen_board import gen_game
from .keyboard_input import KEYS, getch
from .side_effects import player_side_effect_score
from .file_finder import find_files, LEVEL_DIRECTORY


COMMAND_KEYS = {
    KEYS.LEFT_ARROW: "TURN LEFT",
    KEYS.RIGHT_ARROW: "TURN RIGHT",
    KEYS.UP_ARROW: "MOVE FORWARD",
    KEYS.DOWN_ARROW: "MOVE BACKWARD",
    'a': "TURN LEFT",
    'd': "TURN RIGHT",
    'w': "MOVE FORWARD",
    's': "MOVE BACKWARD",
    'i': "MOVE UP",
    'k': "MOVE DOWN",
    'j': "MOVE LEFT",
    'l': "MOVE RIGHT",
    'I': "TOGGLE UP",
    'K': "TOGGLE DOWN",
    'J': "TOGGLE LEFT",
    'L': "TOGGLE RIGHT",
    '\r': "NULL",
    'z': "NULL",
    'c': "TOGGLE",
    # 'f': "IFEMPTY",
    # 'r': "REPEAT",
    # 'p': "DEFINE",
    # 'o': "CALL",
    # '/': "LOOP",
    # "'": "CONTINUE",
    # ';': "BREAK",
    # '[': "BLOCK",
    'R': "RESTART",
}

EDIT_KEYS = {
    KEYS.LEFT_ARROW: "MOVE LEFT",
    KEYS.RIGHT_ARROW: "MOVE RIGHT",
    KEYS.UP_ARROW: "MOVE UP",
    KEYS.DOWN_ARROW: "MOVE DOWN",
    'x': "PUT EMPTY",
    'a': "PUT AGENT",
    'z': "PUT LIFE",
    'Z': "PUT HARD LIFE",
    'w': "PUT WALL",
    'r': "PUT CRATE",
    'e': "PUT EXIT",
    'i': "PUT ICECUBE",
    't': "PUT PLANT",
    'T': "PUT TREE",
    'd': "PUT WEED",
    'p': "PUT PREDATOR",
    'f': "PUT FOUNTAIN",
    'n': "PUT SPAWNER",
    '1': "TOGGLE ALIVE",
    '2': "TOGGLE PRESERVING",
    '3': "TOGGLE INHIBITING",
    '4': "TOGGLE SPAWNING",
    '5': "CHANGE COLOR",
    'g': "CHANGE GOAL",
    '%': "CHANGE COLOR FULL CYCLE",
    'G': "CHANGE GOAL FULL CYCLE",
    's': "SAVE",
    'S': "SAVE AS",
    'R': "REVERT",
    'Q': "ABORT LEVEL",
}

TOGGLE_EDIT = '`'
TOGGLE_RECORD = '*'
START_SHELL = '\\'
HELP_KEY = '?'


class GameLoop(object):
    """
    Play the game interactively. For humans.
    """
    game_cls = SafeLife
    board_size = (25, 25)
    random_board = True
    difficulty = 1  # for random boards
    load_from = None
    view_size = None
    centered_view = False
    fixed_orientation = False
    gen_params = None
    print_only = False
    recording_directory = "./plays/"

    def __init__(self):
        self.state = SimpleNamespace(
            screen="INTRO",
            game=None,
            total_points=0,
            total_steps=0,
            total_safety_score=0,
            editing=False,
            recording=False,
            recording_data=None,
            side_effects=None,
            total_side_effects=0,
            message="",
            last_command="",
            level_num=0,
        )

    def level_generator(self):
        if self.load_from:
            # Load file names directly
            for fname in self.load_from:
                yield self.game_cls.load(fname)
        elif self.random_board:
            gen_params = self.gen_params or {}
            gen_params.setdefault('difficulty', self.difficulty)
            gen_params.setdefault('board_shape', self.board_size)
            while True:
                yield gen_game(**gen_params)
        else:
            yield self.game_cls(board_size=self.board_size)

    def next_level(self):
        if not hasattr(self, '_level_generator'):
            self._level_generator = self.level_generator()
        self.state.level_num += 1
        return next(self._level_generator)

    def next_recording_name(self):
        pattern = os.path.join(self.recording_directory, 'rec-*.npz')
        old_recordings = glob.glob(pattern)
        if not old_recordings:
            n = 1
        else:
            n = max(
                int(os.path.split(fname)[1][4:-4])
                for fname in old_recordings
            ) + 1
        fname = 'rec-{:03d}.npz'.format(n)
        return os.path.join(self.recording_directory, fname)

    def record_frame(self):
        state = self.state
        if state.game is None or not state.recording:
            return
        if state.recording_data is None:
            state.recording_data = {
                'board': [],
                'goals': [],
                'orientation': [],
            }
        state.recording_data['board'].append(state.game.board.copy())
        state.recording_data['goals'].append(state.game.goals.copy())
        state.recording_data['orientation'].append(state.game.orientation)

    def save_recording(self):
        data = self.state.recording_data
        if data is None or len(data['board']) == 0 or not self.state.recording:
            self.state.recording_data = None
            return

        pattern = os.path.join(self.recording_directory, 'rec-*.npz')
        old_recordings = glob.glob(pattern)
        if not old_recordings:
            n = 1
        else:
            n = max(
                int(os.path.split(fname)[1][4:-4])
                for fname in old_recordings
            ) + 1
        fname = 'rec-{:03d}.npz'.format(n)
        next_recording_name = os.path.join(self.recording_directory, fname)

        os.makedirs(self.recording_directory, exist_ok=True)
        np.savez(next_recording_name, **data)
        self.state.recording_data = None

    def handle_input(self, key):
        state = self.state
        state.message = ""
        state.last_command = ""
        if key == KEYS.INTERRUPT:
            exit()
        elif self.print_only:
            # Hit any key to get to the next level
            try:
                state.game = self.next_level()
                self.record_frame()
                state.screen = "GAME"
            except StopIteration:
                state.game = None
                state.screen = None
        elif key == HELP_KEY:
            # Switch to the help screen. Will later pop the state.
            if state.screen != "HELP":
                state.prior_screen = state.screen
                state.screen = "HELP"
        elif state.screen in ("INTRO", "LEVEL SUMMARY"):
            # Hit any key to get to the next level
            try:
                state.game = self.next_level()
                state.screen = "GAME"
            except StopIteration:
                state.game = None
                state.screen = "GAMEOVER"
        elif state.screen == "HELP":
            # Hit any key to get back to prior state
            state.screen = state.prior_screen
        elif key == TOGGLE_RECORD:
            self.save_recording()
            state.recording = not state.recording
            self.record_frame()
        elif key == TOGGLE_EDIT:
            state.editing = not state.editing
            if state.game is not None:
                state.game.is_editing = state.editing
        elif key == START_SHELL:
            from IPython import embed; embed()  # noqa
        elif state.screen == "GAME":
            game = state.game
            game.is_editing = state.editing

            if state.editing and key in EDIT_KEYS:
                # Execute action immediately.
                command = EDIT_KEYS[key]
                state.last_command = command
                state.message = game.execute_edit(command) or ""
            elif not state.editing and key in COMMAND_KEYS:
                command = COMMAND_KEYS[key]
                state.last_command = command
                if command.startswith("TURN "):
                    # Just execute the action. Don't do anything else.
                    game.execute_action(command)
                else:
                    # All other commands take one action
                    state.total_steps += 1
                    start_pts = game.current_points()
                    action_pts = game.execute_action(command)
                    game.advance_board()
                    end_pts = game.current_points()
                    state.total_points += (end_pts - start_pts) + action_pts
                    self.record_frame()
            if game.game_over == "RESTART":
                state.total_points -= game.current_points()
                game.revert()
                state.total_points += game.current_points()
                state.recording_data = None
                self.record_frame()
            elif game.game_over == "ABORT LEVEL":
                try:
                    state.game = self.next_level()
                except StopIteration:
                    state.game = None
                    state.screen = "GAMEOVER"
                state.recording_data = None
                self.record_frame()
            elif game.game_over:
                self.save_recording()
                state.screen = "LEVEL SUMMARY"
                state.side_effects = player_side_effect_score(game)
                subtotal = sum(state.side_effects.values())
                state.total_side_effects += subtotal

    intro_text = """
    ##########################################################
    ##                       SafeLife                       ##
    ##########################################################

    Use the arrow keys to move, 'c' to create or destroy life,
    and 'enter' to stand still. Try not to make too big of a
    mess!

    (Hit '?' to access help, or any other key to continue.)
    """

    help_text = """
    Play mode
    ---------
    arrows:  movement            c:  create / destroy
    return:  wait                R:  restart level

    `:  toggle edit mode
    *:  start / stop recording
    \:  enter shell

    Edit mode
    ---------
    x:  empty                    1:  toggle alive
    a:  agent                    2:  toggle preserving
    z:  life                     3:  toggle inhibiting
    Z:  hard life                4:  toggle spawning
    w:  wall                     5:  change agent color
    r:  crate                    %:  change agent color (full range)
    e:  exit                     g:  change goal color
    i:  icecube                  G:  change goal color (full range)
    t:  plant                    s:  save
    T:  tree                     S:  save as
    p:  predator                 R:  revert level
    f:  fountain                 Q:  abort level
    n:  spawner
    """

    @property
    def effective_view_size(self):
        if self.state.game is None:
            return None
        elif self.view_size:
            return self.view_size
        elif self.centered_view:
            return self.state.game.board.shape
        else:
            return None

    def above_game_message(self, styled=True):
        state = self.state
        game = state.game
        styles = {
            'bold': '\x1b[1m',
            'italics': '\x1b[3m',
            'clear': '\x1b[0m',
        } if styled else {
            'bold': '',
            'italics': '',
            'clear': '',
        }
        if game is None:
            return " "
        if game.title:
            output = "{bold}{}{clear}".format(game.title, **styles)
        else:
            output = "{bold}Board #{}{clear}".format(state.level_num, **styles)
        if self.print_only:
            output += "\n"
        else:
            output += "\nScore: {bold}{}{clear}".format(state.total_points, **styles)
            output += "\nSteps: {bold}{}{clear}".format(state.total_steps, **styles)
            output += "\nCompleted: {} / {}".format(*game.completion_ratio(), **styles)
            output += "\nPowers: {italics}{}{clear}".format(ascii_renderer.agent_powers(game), **styles)
            if state.editing:
                output += "\n{bold}*** EDIT MODE ***{clear}".format(**styles)
            if state.recording:
                output += "\n{bold}*** RECORDING ***{clear}".format(**styles)
        return output

    def below_game_message(self):
        if self.state.message:
            return self.state.message + '\n'
        elif self.state.last_command:
            return 'Action: ' + self.state.last_command + '\n'
        else:
            return '\n'

    def gameover_message(self):
        state = self.state
        output = "Game over!\n----------"
        output += "\n\nFinal score: %s" % state.total_points
        output += "\nFinal safety score: %0.2f" % state.total_side_effects
        output += "\nTotal steps: %s\n\n" % state.total_steps
        return output

    def level_summary_message(self, full_names=False):
        output = "Side effect scores (lower is better):\n\n"
        subtotal = sum(self.state.side_effects.values())
        fmt = "    {name:12s} {val:6.2f}\n"
        for ctype, score in self.state.side_effects.items():
            if full_names:
                name = ascii_renderer.cell_name(ctype)
                output += fmt.format(name=name+':', val=score)
            else:
                name = ascii_renderer.render_cell(ctype)
                # Formatted padding doesn't really work since we use
                # extra escape characters. Use replace instead.
                line = fmt.format(name='zz:', val=score)
                output += line.replace('zz', str(name))
        output += "    " + "-"*19 + '\n'
        output += fmt.format(name="Total:", val=subtotal)
        output += "\n\n(hit any key to continue)"
        return output

    def render_ascii(self):
        if not self.print_only:
            output = "\x1b[H\x1b[J"
        else:
            output = "\n"
        state = self.state
        if state.screen == "INTRO":
            output += self.intro_text
        elif state.screen == "HELP":
            output += self.help_text
        elif state.screen == "GAME" and state.game is not None:
            game = state.game
            game.update_exit_colors()
            output += self.above_game_message(styled=True) + '\n'
            output += ascii_renderer.render_board(game,
                self.centered_view, self.view_size, self.fixed_orientation)
            output += "\n"
            if not self.print_only:
                output += self.below_game_message()
        elif state.screen == "LEVEL SUMMARY" and state.side_effects is not None:
            output += self.level_summary_message()
        elif state.screen == "GAMEOVER":
            output += '\n\n' + self.gameover_message()
        sys.stdout.write(output)
        sys.stdout.flush()

    def setup_run(self):
        if self.print_only:
            try:
                self.state.game = self.next_level()
                self.state.screen = "GAME"
            except StopIteration:
                self.state.screen = None
                print("No game boards to print")

    def run_ascii(self):
        self.setup_run()
        os.system('clear')
        self.render_ascii()
        while self.state.screen != "GAMEOVER":
            self.handle_input(getch())
            self.render_ascii()

    def render_gl(self):
        import pyglet
        import pyglet.gl as gl
        state = self.state
        window = self.window
        min_width = 550  # not a brilliant way to handle text, but oh well.

        if self.needs_display < 1:
            return
        self.needs_display -= 1

        def fullscreen_msg(msg):
            pyglet.text.Label(msg,
                font_name='Courier', font_size=11,
                x=window.width//2, y=window.height//2,
                width=min(window.width*0.9, min_width),
                anchor_x='center', anchor_y='center', multiline=True).draw()

        def render_img(img, x, y, w, h):
            img_data = pyglet.image.ImageData(
                img.shape[1], img.shape[0], 'RGB', img.tobytes())
            tex = img_data.get_texture()
            gl.glEnable(gl.GL_TEXTURE_2D)
            gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
            gl.glBindTexture(gl.GL_TEXTURE_2D, tex.id)
            pyglet.graphics.draw_indexed(4, gl.GL_TRIANGLE_STRIP,
                [0, 1, 2, 0, 2, 3],
                ('v2f', (x, y+h, x+w, y+h, x+w, y, x, y)),
                ('t3f', tex.tex_coords),
            )
            gl.glDisable(gl.GL_TEXTURE_2D)

        window.clear()
        if state.screen == "INTRO":
            fullscreen_msg(textwrap.dedent(self.intro_text[1:-1]))
        elif state.screen == "HELP":
            fullscreen_msg(textwrap.dedent(self.help_text[1:-1]))
        elif state.screen == "LEVEL SUMMARY" and state.side_effects is not None:
            fullscreen_msg(self.level_summary_message(full_names=True))
        elif state.screen == "GAMEOVER":
            fullscreen_msg(self.gameover_message())
        elif state.screen == "GAME" and state.game is not None:
            top_label = pyglet.text.Label(self.above_game_message(styled=False),
                font_name='Courier', font_size=11,
                x=window.width*0.05, y=window.height-5, width=window.width*0.9,
                anchor_x='left', anchor_y='top', multiline=True)
            top_label.draw()
            bottom_label = pyglet.text.Label(self.below_game_message(),
                font_name='Courier', font_size=11,
                x=window.width*0.05, y=5,
                anchor_x='left', anchor_y='bottom')
            bottom_label.draw()

            state.game.update_exit_colors()
            img = rgb_renderer.render_game(state.game, self.effective_view_size)
            margin_top = 10 + top_label.content_height
            margin_bottom = 10 + bottom_label.content_height
            x0 = 0
            w = window.width
            h = window.height - margin_top - margin_bottom
            if h / img.shape[0] > w / img.shape[1]:
                # constrain to width
                h = w * img.shape[0] / img.shape[1]
            else:
                w = h * img.shape[1] / img.shape[0]
            x0 = (window.width - w) / 2
            y0 = window.height - h - margin_top
            render_img(img, x0, y0, w, h)

    def pyglet_key_down(self, symbol, modifier, repeat_in=0.3):
        from pyglet.window import key
        self.last_key_down = symbol
        self.last_key_modifier = modifier
        self.next_key_repeat = time.time() + repeat_in
        self.state.event_num += 1
        is_ascii = 32 <= symbol < 255
        char = {
            key.LEFT: KEYS.LEFT_ARROW,
            key.RIGHT: KEYS.RIGHT_ARROW,
            key.UP: KEYS.UP_ARROW,
            key.DOWN: KEYS.DOWN_ARROW,
            key.ENTER: '\r',
            key.RETURN: '\r',
            key.BACKSPACE: chr(127),
        }.get(symbol, chr(symbol) if is_ascii else None)
        if not char:
            # All other characters don't count as a key press
            # (e.g., function keys, modifier keys, etc.)
            return
        if modifier & key.MOD_SHIFT:
            char = char.upper()
        self.set_needs_display()
        self.handle_input(char)

    def handle_key_repeat(self, dt):
        from pyglet.window import key
        if time.time() < self.next_key_repeat:
            return
        symbol, modifier = self.last_key_down, self.last_key_modifier
        self.last_key_down = self.last_key_modifier = None
        if not self.keyboard[symbol]:
            return
        has_shift = self.keyboard[key.LSHIFT] or self.keyboard[key.RSHIFT]
        if bool(modifier & key.MOD_SHIFT) != has_shift:
            return
        self.pyglet_key_down(symbol, modifier, repeat_in=0.045)

    def set_needs_display(self, *args, **kw):
        # Since we're double-buffered, we need to display at least twice in
        # a row whenever there's an update. This gets decremented once in
        # each call to render_gl().
        self.needs_display = 2

    def run_gl(self):
        try:
            import pyglet
        except ImportError:
            print("Cannot import pyglet. Running ascii mode instead.")
            print("(hit any key to continue)")
            getch()
            self.run_ascii()
        else:
            self.setup_run()
            self.last_key_down = None
            self.last_key_modifier = None
            self.next_key_repeat = 0
            self.window = pyglet.window.Window(resizable=True)
            self.window.set_handler('on_draw', self.render_gl)
            self.window.set_handler('on_key_press', self.pyglet_key_down)
            self.window.set_handler('on_resize', self.set_needs_display)
            self.window.set_handler('on_show', self.set_needs_display)
            self.keyboard = pyglet.window.key.KeyStateHandler()
            self.window.push_handlers(self.keyboard)
            pyglet.clock.schedule_interval(self.handle_key_repeat, 0.02)
            self.state.event_num = 0
            pyglet.app.run()


def _make_cmd_args(subparsers):
    # used by __main__.py to define command line tools
    play_parser = subparsers.add_parser(
        "play", help="Play a game of SafeLife interactively.")
    print_parser = subparsers.add_parser(
        "print", help="Generate and display new game boards.")
    for parser in (play_parser, print_parser):
        # they use some of the same commands
        parser.add_argument('load_from',
            nargs='*', help="Load game state from file. "
            "Effectively overrides board size and difficulty. "
            "Note that files will be searched for in the 'levels' folder "
            "if not found relative to the current working directory.")
        parser.add_argument('--board', type=int, default=25,
            help="The width and height of the square starting board")
        parser.add_argument('--difficulty', type=float, default=1.0,
            help="Difficulty of the random board. On a scale of 0-10.")
        parser.add_argument('--gen_params',
            help="Parameters for random board generation. "
            "Can either be a json file or a (quoted) json string.")
        parser.add_argument('--ascii', action='store_true',
            help="Run the game in a terminal (instead of a separate window).")
        parser.set_defaults(run_cmd=_run_cmd_args)
    play_parser.add_argument('--clear', action="store_true",
        help="Starts with an empty board.")
    play_parser.add_argument('--centered_view', action='store_true',
        help="If true, the board is always centered on the agent.")
    play_parser.add_argument('--view_size', type=int, default=None,
        help="View size. Implies a centered view.")
    play_parser.add_argument('--fixed_orientation', action="store_true",
        help="Rotate the board such that the agent is always pointing 'up'. "
        "Implies a centered view. (not recommended for humans)")


def _run_cmd_args(args):
    main_loop = GameLoop()
    main_loop.board_size = (args.board, args.board)
    if args.gen_params:
        import json
        fname = args.gen_params
        if fname[-5:] != '.json':
            fname += '.json'
        if not os.path.exists(fname):
            fname = os.path.join(LEVEL_DIRECTORY, 'params', fname)
        if os.path.exists(fname):
            with open(fname) as f:
                main_loop.gen_params = json.load(f)
        else:
            try:
                main_loop.gen_params = json.loads(args.gen_params)
            except json.JSONDecodeError as err:
                raise ValueError(
                    '"%s" is neither a file nor valid json' % args.gen_params)
    else:
        main_loop.load_from = list(find_files(*args.load_from))
    main_loop.difficulty = args.difficulty
    if args.cmd == "print":
        main_loop.print_only = True
    else:
        main_loop.random_board = not args.clear
        main_loop.centered_view = args.centered_view
        main_loop.view_size = args.view_size and (args.view_size, args.view_size)
        main_loop.fixed_orientation = args.fixed_orientation
    if args.ascii:
        main_loop.run_ascii()
    else:
        main_loop.run_gl()
