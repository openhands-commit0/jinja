"""Compiles nodes from the parser into Python code."""
import typing as t
from contextlib import contextmanager
from functools import update_wrapper
from io import StringIO
from itertools import chain
from keyword import iskeyword as is_python_keyword
from markupsafe import escape
from markupsafe import Markup
from . import nodes
from .exceptions import TemplateAssertionError
from .idtracking import Symbols
from .idtracking import VAR_LOAD_ALIAS
from .idtracking import VAR_LOAD_PARAMETER
from .idtracking import VAR_LOAD_RESOLVE
from .idtracking import VAR_LOAD_UNDEFINED
from .nodes import EvalContext
from .optimizer import Optimizer
from .utils import _PassArg
from .utils import concat
from .visitor import NodeVisitor
if t.TYPE_CHECKING:
    import typing_extensions as te
    from .environment import Environment
F = t.TypeVar('F', bound=t.Callable[..., t.Any])
operators = {'eq': '==', 'ne': '!=', 'gt': '>', 'gteq': '>=', 'lt': '<', 'lteq': '<=', 'in': 'in', 'notin': 'not in'}

def generate(node: nodes.Template, environment: 'Environment', name: t.Optional[str], filename: t.Optional[str], stream: t.Optional[t.TextIO]=None, defer_init: bool=False, optimized: bool=True) -> t.Optional[str]:
    """Generate the python source for a node tree."""
    pass

def has_safe_repr(value: t.Any) -> bool:
    """Does the node have a safe representation?"""
    pass

def find_undeclared(nodes: t.Iterable[nodes.Node], names: t.Iterable[str]) -> t.Set[str]:
    """Check if the names passed are accessed undeclared.  The return value
    is a set of all the undeclared names from the sequence of names found.
    """
    pass

class MacroRef:

    def __init__(self, node: t.Union[nodes.Macro, nodes.CallBlock]) -> None:
        self.node = node
        self.accesses_caller = False
        self.accesses_kwargs = False
        self.accesses_varargs = False

class Frame:
    """Holds compile time information for us."""

    def __init__(self, eval_ctx: EvalContext, parent: t.Optional['Frame']=None, level: t.Optional[int]=None) -> None:
        self.eval_ctx = eval_ctx
        self.parent = parent
        if parent is None:
            self.symbols = Symbols(level=level)
            self.require_output_check = False
            self.buffer: t.Optional[str] = None
            self.block: t.Optional[str] = None
        else:
            self.symbols = Symbols(parent.symbols, level=level)
            self.require_output_check = parent.require_output_check
            self.buffer = parent.buffer
            self.block = parent.block
        self.toplevel = False
        self.rootlevel = False
        self.loop_frame = False
        self.block_frame = False
        self.soft_frame = False

    def copy(self) -> 'Frame':
        """Create a copy of the current one."""
        rv = object.__new__(Frame)
        rv.eval_ctx = self.eval_ctx
        rv.parent = self.parent
        rv.symbols = self.symbols
        rv.require_output_check = self.require_output_check
        rv.buffer = self.buffer
        rv.block = self.block
        rv.toplevel = self.toplevel
        rv.rootlevel = self.rootlevel
        rv.loop_frame = self.loop_frame
        rv.block_frame = self.block_frame
        rv.soft_frame = self.soft_frame
        return rv

    def inner(self, isolated: bool=False) -> 'Frame':
        """Return an inner frame."""
        rv = Frame(self.eval_ctx, self, level=self.symbols.level + 1)
        if isolated:
            rv.symbols.add_level()
        return rv

    def soft(self) -> 'Frame':
        """Return a soft frame.  A soft frame may not be modified as
        standalone thing as it shares the resources with the frame it
        was created of, but it's not a rootlevel frame any longer.

        This is only used to implement if-statements and conditional
        expressions.
        """
        rv = object.__new__(Frame)
        rv.eval_ctx = self.eval_ctx
        rv.parent = self.parent
        rv.symbols = self.symbols
        rv.require_output_check = self.require_output_check
        rv.buffer = self.buffer
        rv.block = self.block
        rv.toplevel = False
        rv.rootlevel = False
        rv.loop_frame = self.loop_frame
        rv.block_frame = self.block_frame
        rv.soft_frame = True
        return rv
    __copy__ = copy

class VisitorExit(RuntimeError):
    """Exception used by the `UndeclaredNameVisitor` to signal a stop."""

class DependencyFinderVisitor(NodeVisitor):
    """A visitor that collects filter and test calls."""

    def __init__(self) -> None:
        self.filters: t.Set[str] = set()
        self.tests: t.Set[str] = set()

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting at blocks."""
        return

class UndeclaredNameVisitor(NodeVisitor):
    """A visitor that checks if a name is accessed without being
    declared.  This is different from the frame visitor as it will
    not stop at closure frames.
    """

    def __init__(self, names: t.Iterable[str]) -> None:
        self.names = set(names)
        self.undeclared: t.Set[str] = set()

    def visit_Block(self, node: nodes.Block) -> None:
        """Stop visiting a blocks."""
        return

class CompilerExit(Exception):
    """Raised if the compiler encountered a situation where it just
    doesn't make sense to further process the code.  Any block that
    raises such an exception is not further processed.
    """

def _make_binop(op: str) -> t.Callable[['CodeGenerator', nodes.BinExpr, Frame], None]:
    def visitor(self: 'CodeGenerator', node: nodes.BinExpr, frame: Frame) -> None:
        self.write('(')
        self.visit(node.left, frame)
        self.write(f' {op} ')
        self.visit(node.right, frame)
        self.write(')')
    return visitor

def _make_unop(op: str) -> t.Callable[['CodeGenerator', nodes.UnaryExpr, Frame], None]:
    def visitor(self: 'CodeGenerator', node: nodes.UnaryExpr, frame: Frame) -> None:
        self.write('(' + op)
        self.visit(node.node, frame)
        self.write(')')
    return visitor

class CodeGenerator(NodeVisitor):

    def __init__(self, environment: 'Environment', name: t.Optional[str], filename: t.Optional[str], stream: t.Optional[t.TextIO]=None, defer_init: bool=False, optimized: bool=True) -> None:
        if stream is None:
            stream = StringIO()
        self.environment = environment
        self.name = name
        self.filename = filename
        self.stream = stream
        self.created_block_context = False
        self.defer_init = defer_init
        self.optimizer: t.Optional[Optimizer] = None
        if optimized:
            self.optimizer = Optimizer(environment)
        self.import_aliases: t.Dict[str, str] = {}
        self.blocks: t.Dict[str, nodes.Block] = {}
        self.extends_so_far = 0
        self.has_known_extends = False
        self.code_lineno = 1
        self.tests: t.Dict[str, str] = {}
        self.filters: t.Dict[str, str] = {}
        self.debug_info: t.List[t.Tuple[int, int]] = []
        self._write_debug_info: t.Optional[int] = None
        self._new_lines = 0
        self._last_line = 0
        self._first_write = True
        self._last_identifier = 0
        self._indentation = 0
        self._assign_stack: t.List[t.Set[str]] = []
        self._param_def_block: t.List[t.Set[str]] = []
        self._context_reference_stack = ['context']

    def fail(self, msg: str, lineno: int) -> 'te.NoReturn':
        """Fail with a :exc:`TemplateAssertionError`."""
        raise TemplateAssertionError(msg, lineno, self.name, self.filename)

    def temporary_identifier(self) -> str:
        """Get a new unique identifier."""
        self._last_identifier += 1
        return f'_tmp_{self._last_identifier}'

    def buffer(self, frame: Frame) -> None:
        """Enable buffering for the frame from that point onwards."""
        frame.buffer = self.temporary_identifier()

    def return_buffer_contents(self, frame: Frame, force_unescaped: bool=False) -> None:
        """Return the buffer contents of the frame."""
        if not frame.buffer:
            self.writeline('if 0: yield None')
            return

        if force_unescaped:
            self.writeline(f'yield str({frame.buffer})')
        elif frame.eval_ctx.volatile:
            self.writeline(f'yield str({frame.buffer}) if context.eval_ctx.autoescape else {frame.buffer}')
        elif frame.eval_ctx.autoescape:
            self.writeline(f'yield str({frame.buffer})')
        else:
            self.writeline(f'yield {frame.buffer}')

    def indent(self) -> None:
        """Indent by one."""
        self._indentation += 1

    def outdent(self, step: int=1) -> None:
        """Outdent by step."""
        self._indentation = max(0, self._indentation - step)

    def start_write(self, frame: Frame, node: t.Optional[nodes.Node]=None) -> None:
        """Yield or write into the frame buffer."""
        if frame.buffer is None:
            self.writeline('yield ', node)
        else:
            self.writeline(f'{frame.buffer} = ', node)

    def end_write(self, frame: Frame) -> None:
        """End the writing process started by `start_write`."""
        if frame.buffer is not None:
            self.writeline(f'yield {frame.buffer}')

    def simple_write(self, s: str, frame: Frame, node: t.Optional[nodes.Node]=None) -> None:
        """Simple shortcut for start_write + write + end_write."""
        self.start_write(frame, node)
        self.write(s)
        self.end_write(frame)

    def blockvisit(self, nodes: t.Iterable[nodes.Node], frame: Frame) -> None:
        """Visit a list of nodes as block in a frame.  If the current frame
        is no buffer a dummy ``if 0: yield None`` is written automatically.
        """
        if frame.buffer is None:
            self.writeline('if 0: yield None')
        for node in nodes:
            self.visit(node, frame)

    def write(self, x: str) -> None:
        """Write a string into the output stream."""
        if self._new_lines:
            if not self._first_write:
                self.stream.write('\n' * self._new_lines)
            self.stream.write('    ' * self._indentation)
            self._new_lines = 0
        self.stream.write(x)
        self._first_write = False

    def writeline(self, x: str, node: t.Optional[nodes.Node]=None, extra: int=0) -> None:
        """Combination of newline and write."""
        if node is not None:
            self._write_debug_info(node)
        self.newline(node, extra)
        self.write(x)

    def newline(self, node: t.Optional[nodes.Node]=None, extra: int=0) -> None:
        """Add one or more newlines before the next write."""
        if node is not None:
            self._write_debug_info(node)
        self._new_lines = max(self._new_lines, 1 + extra)

    def signature(self, node: t.Union[nodes.Call, nodes.Filter, nodes.Test], frame: Frame, extra_kwargs: t.Optional[t.Mapping[str, t.Any]]=None) -> None:
        """Writes a function call to the stream for the current node.
        A leading comma is added automatically.  The extra keyword
        arguments may not include python keywords otherwise a syntax
        error could occur.  The extra keyword arguments should be given
        as python dict.
        """
        if extra_kwargs is None:
            extra_kwargs = {}

        # Regular argument handling
        for arg in node.args:
            self.write(', ')
            self.visit(arg, frame)

        # Keyword argument handling
        for kwarg in node.kwargs:
            self.write(', ')
            self.write(kwarg.key + '=')
            self.visit(kwarg.value, frame)

        # Dynamic arguments
        if node.dyn_args is not None:
            self.write(', *')
            self.visit(node.dyn_args, frame)

        # Dynamic keyword arguments
        if node.dyn_kwargs is not None:
            self.write(', **')
            self.visit(node.dyn_kwargs, frame)

        # Extra keyword arguments from the call
        for key, value in extra_kwargs.items():
            self.write(f', {key}={value!r}')

    def pull_dependencies(self, nodes: t.Iterable[nodes.Node]) -> None:
        """Find all filter and test names used in the template and
        assign them to variables in the compiled namespace. Checking
        that the names are registered with the environment is done when
        compiling the Filter and Test nodes. If the node is in an If or
        CondExpr node, the check is done at runtime instead.

        .. versionchanged:: 3.0
            Filters and tests in If and CondExpr nodes are checked at
            runtime instead of compile time.
        """
        visitor = DependencyFinderVisitor()
        for node in nodes:
            visitor.visit(node)
        for name in visitor.filters:
            if name not in self.filters:
                self.filters[name] = self.temporary_identifier()
        for name in visitor.tests:
            if name not in self.tests:
                self.tests[name] = self.temporary_identifier()

    def macro_body(self, node: t.Union[nodes.Macro, nodes.CallBlock], frame: Frame) -> t.Tuple[Frame, MacroRef]:
        """Dump the function def of a macro or call block."""
        macro_ref = MacroRef(node)
        body_frame = frame.inner()
        body_frame.require_output_check = False
        self.writeline('def macro(', node)
        self.indent()
        self.write('caller=None')
        for arg in node.args:
            self.write(', ' + arg.name)
            if arg.default is not None:
                self.write('=')
                self.visit(arg.default, frame)
        self.write(', kwargs=None')
        self.write('):')
        self.indent()
        self.buffer(body_frame)
        self.pull_dependencies([node])
        self.blockvisit(node.body, body_frame)
        self.return_buffer_contents(body_frame)
        self.outdent(2)
        return body_frame, macro_ref

    def macro_def(self, macro_ref: MacroRef, frame: Frame) -> None:
        """Dump the macro definition for the def created by macro_body."""
        self.write('Macro(environment, macro, None, ')
        if macro_ref.accesses_kwargs:
            self.write('True')
        else:
            self.write('False')
        if macro_ref.accesses_varargs:
            self.write(', True')
        else:
            self.write(', False')
        if macro_ref.accesses_caller:
            self.write(', True')
        else:
            self.write(', False')
        self.write(')')

    def position(self, node: nodes.Node) -> str:
        """Return a human readable position for the node."""
        rv = f'line {node.lineno}'
        if self.name is not None:
            rv = f'{rv} in {self.name}'
        return rv

    def write_commons(self) -> None:
        """Writes a common preamble that is used by root and block functions.
        Primarily this sets up common local helpers and enforces a generator
        through a dead branch.
        """
        self.writeline('resolve = context.resolve_or_missing')
        self.writeline('undefined = environment.undefined')
        self.writeline('if 0: yield None')

    def push_parameter_definitions(self, frame: Frame) -> None:
        """Pushes all parameter targets from the given frame into a local
        stack that permits tracking of yet to be assigned parameters.  In
        particular this enables the optimization from `visit_Name` to skip
        undefined expressions for parameters in macros as macros can reference
        otherwise unbound parameters.
        """
        self._param_def_block.append(frame.symbols.stores)

    def pop_parameter_definitions(self) -> None:
        """Pops the current parameter definitions set."""
        self._param_def_block.pop()

    def mark_parameter_stored(self, target: str) -> None:
        """Marks a parameter in the current parameter definitions as stored.
        This will skip the enforced undefined checks.
        """
        if self._param_def_block:
            self._param_def_block[-1].add(target)

    def parameter_is_undeclared(self, target: str) -> bool:
        """Checks if a given target is an undeclared parameter."""
        if not self._param_def_block:
            return False
        return target not in self._param_def_block[-1]

    def push_assign_tracking(self) -> None:
        """Pushes a new layer for assignment tracking."""
        self._assign_stack.append(set())

    def pop_assign_tracking(self, frame: Frame) -> None:
        """Pops the topmost level for assignment tracking and updates the
        context variables if necessary.
        """
        assignments = self._assign_stack.pop()
        if assignments:
            self.writeline(f'{frame.symbols.store_set(assignments)} = context.exported_vars.add_update({assignments!r})')

    def visit_Block(self, node: nodes.Block, frame: Frame) -> None:
        """Call a block and register it for the template."""
        block_frame = frame.inner()
        block_frame.block = node.name
        self.writeline(f'def block_{node.name}(context, missing=missing):', node)
        self.indent()
        self.buffer(block_frame)
        self.blockvisit(node.body, block_frame)
        self.return_buffer_contents(block_frame)
        self.outdent()
        self.blocks[node.name] = node
        self.writeline(f'blocks[{node.name!r}] = block_{node.name}')

    def visit_Extends(self, node: nodes.Extends, frame: Frame) -> None:
        """Calls the extender."""
        if not frame.toplevel:
            self.fail('cannot use extend from a non top-level scope', node.lineno)
        if self.has_known_extends:
            self.fail('cannot have multiple extends tags', node.lineno)
        if not self.extends_so_far:
            self.writeline('parent_template = None')
        self.writeline('for parent in [', node)
        self.visit(node.template, frame)
        self.write(']:')
        self.indent()
        self.writeline('parent_template = parent')
        self.writeline('break')
        self.outdent()
        self.has_known_extends = True

    def visit_Include(self, node: nodes.Include, frame: Frame) -> None:
        """Handles includes."""
        if node.ignore_missing:
            self.writeline('try:')
            self.indent()
        self.writeline('for event in (', node)
        if node.with_context:
            self.write('context.blocks[')
        else:
            self.write('environment.get_template(')
        self.visit(node.template, frame)
        if node.ignore_missing:
            self.write(').root_render_func(context)')
            self.outdent()
            self.writeline('except TemplateNotFound:')
            self.indent()
            self.writeline('pass')
            self.outdent()
        else:
            self.write(').root_render_func(context)')
            self.write('):')
            self.indent()
            self.writeline('yield event')
            self.outdent()

    def visit_Import(self, node: nodes.Import, frame: Frame) -> None:
        """Visit regular imports."""
        self.writeline(f'template = environment.get_template(', node)
        self.visit(node.template, frame)
        self.write(')')
        self.writeline(f'{frame.symbols.ref(node.target)} = environment.make_module(')
        self.write('template.make_module(context.get_all()' if node.with_context else 'template.make_module()')
        self.write(')')
        frame.symbols.store(node.target)

    def visit_FromImport(self, node: nodes.FromImport, frame: Frame) -> None:
        """Visit named imports."""
        self.writeline('template = environment.get_template(', node)
        self.visit(node.template, frame)
        self.write(')')
        self.writeline('module = template.make_module(' + ('context.get_all()' if node.with_context else '') + ')')
        for name in node.names:
            if isinstance(name, tuple):
                name, alias = name
            else:
                alias = name
            self.writeline(f'{frame.symbols.ref(alias)} = getattr(module, {name!r}, missing)')
            frame.symbols.store(alias)

    class _FinalizeInfo(t.NamedTuple):
        const: t.Optional[t.Callable[..., str]]
        src: t.Optional[str]

    @staticmethod
    def _default_finalize(value: t.Any) -> t.Any:
        """The default finalize function if the environment isn't
        configured with one. Or, if the environment has one, this is
        called on that function's output for constants.
        """
        return value
    _finalize: t.Optional[_FinalizeInfo] = None

    def _make_finalize(self) -> _FinalizeInfo:
        """Build the finalize function to be used on constants and at
        runtime. Cached so it's only created once for all output nodes.

        Returns a ``namedtuple`` with the following attributes:

        ``const``
            A function to finalize constant data at compile time.

        ``src``
            Source code to output around nodes to be evaluated at
            runtime.
        """
        finalize = self.environment.finalize
        if finalize is None:
            return self._FinalizeInfo(const=self._default_finalize, src=None)
        
        src = self.temporary_identifier()
        self.writeline(f'{src} = environment.finalize')
        return self._FinalizeInfo(const=finalize, src=src)

    def _output_const_repr(self, group: t.Iterable[t.Any]) -> str:
        """Given a group of constant values converted from ``Output``
        child nodes, produce a string to write to the template module
        source.
        """
        return repr(concat(group))

    def _output_child_to_const(self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo) -> str:
        """Try to optimize a child of an ``Output`` node by trying to
        convert it to constant, finalized data at compile time.

        If :exc:`Impossible` is raised, the node is not constant and
        will be evaluated at runtime. Any other exception will also be
        evaluated at runtime for easier debugging.
        """
        const = node.as_const(frame.eval_ctx)
        if finalize.const is not None:
            const = finalize.const(const)
        return str(const)

    def _output_child_pre(self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo) -> None:
        """Output extra source code before visiting a child of an
        ``Output`` node.
        """
        if finalize.src is not None:
            self.write(f'{finalize.src}(')

    def _output_child_post(self, node: nodes.Expr, frame: Frame, finalize: _FinalizeInfo) -> None:
        """Output extra source code after visiting a child of an
        ``Output`` node.
        """
        if finalize.src is not None:
            self.write(')')
    visit_Add = _make_binop('+')
    visit_Sub = _make_binop('-')
    visit_Mul = _make_binop('*')
    visit_Div = _make_binop('/')
    visit_FloorDiv = _make_binop('//')
    visit_Pow = _make_binop('**')
    visit_Mod = _make_binop('%')
    visit_And = _make_binop('and')
    visit_Or = _make_binop('or')
    visit_Pos = _make_unop('+')
    visit_Neg = _make_unop('-')
    visit_Not = _make_unop('not ')