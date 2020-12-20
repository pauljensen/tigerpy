
import pyparsing as pp

class Expression(object):
    def is_empty(self):
        return False


class Atom(Expression):
    """Base value in a logical expression.
    
    For a GPR, each gene is an Atom. When printed, Atoms are prefixed with
    a `$`.
    
    """
    def __init__(self, name):
        self.name = name
    
    def atoms(self):
        return [self.name]
    
    def to_rule(self, dictname="x"):
        return dictname + "['" + self.name + "']"
    
    def __str__(self):
        return "$" + self.name
    
    def has(self, op):
        return False
    
    __repr__ = __str__
    
    
class Empty(Expression):
    """An empty rule with no Atoms of Junctions.
    
    Empty can represent either an empty Junction or a missing Atom.
    
    Parsing an empty string (or a string of only whitspaces) returns an
    Empty object. When printed, Empty objects appear as '<empty>'.
    
    """
    def __str__(self):
        return "<empty>"
    
    def atoms(self):
        return []
    
    def to_rule(self, dictname="x"):
        return ''
    
    def has(self, op):
        return False
    
    def is_empty(self):
        return True
    
    __repr__ = __str__
    

class Junction(Expression):
    """A Boolean operator combined with one or more operands.
    
    Junctions are not useful directly. Use And or Or expressions instead.
    
    Junctions have two fields:
        operator [str]
        operands List[Atom | Junction | Empty]
        
    """
    def __init__(self, operator, operands):
        self.operator = operator
        self.operands = operands
        
    def atoms(self):
        return list(set([i for ops in self.operands for i in ops.atoms()]))
    
    def to_rule(self, dictname="x"):
        ops = ['(' + op.to_rule(dictname=dictname) + ')' for op in self.operands]
        return (' ' + self.operator + ' ').join(ops)
        
    def __str__(self):
        return self.operator + "(" + ", ".join(map(str, self.operands)) + ")"
    
    def has(self, op):
        return self.operator == op or any([x.has(op) for x in self.operands])
        
    __repr__ = __str__
        

class And(Junction):
    """Logical And (conjunction)."""
    def __init__(self, operands):
        super().__init__("and", operands)
        

class Or(Junction):
    """Logical Or (disjunction)."""
    def __init__(self, operands):
        super().__init__("or", operands)


def show_tree(expr, indent='', indent_step='  '):
    """Print a logical expression in a tree-like format."""
    if isinstance(expr, Junction):
        print(indent + "+", expr.operator)
        for op in expr.operands:
            show_tree(op, indent=indent + indent_step)
    else:
        print(indent + '|', expr)
  

def pullup_or(expr):
    """Convert a logical expression into DNF form.
    
    Use `make_dnf` instead of calling this function directly.
    
    """
    if isinstance(expr, Or):
        operands = []
        for op in expr.operands:
            operands += pullup_or(op).operands
        return Or(operands)
    elif isinstance(expr, Atom):
        return Or([expr])
    elif isinstance(expr, Empty):
        return Or([])
    else: # And
        operands = [[x] for x in pullup_or(expr.operands[0]).operands]
        for op in expr.operands[1:]:
            new = pullup_or(op).operands
            operands = [x + [y] for x in operands for y in new]
        return Or([And(x) for x in operands])
    
def switch_and_or(expr):
    """Swap And and Or junctions in an expression.
    
    Used to repurpose `pullup_or` to make CNF forms.
    
    Example: And(x, Or(y, z)) -> Or(x, And(y, z))
    
    """
    if isinstance(expr, Or):
        return And([switch_and_or(x) for x in expr.operands])
    elif isinstance(expr, And):
        return Or([switch_and_or(x) for x in expr.operands])
    else:
        return expr
    
def make_dnf(expr, names_only=False):
    """Convert an expression into Disjunctive Normal Form (DNF).
    
    DNF form is Or's of And's, such as
        Or(
            And(x,y),
            z,
            And(y,w)
        )
    
    If `names_only`, returns a list of the atom names in each conjunction:
        [
            ["x","y"],
            ["z"],
            ["y","w"]
        ]
    
    """
    expr = pullup_or(expr)
    if names_only:
        return [x.atoms() for x in expr.operands]
    else:
        return expr

def make_cnf(expr, names_only=False):
    """Convert an expression into Conjunctive Normal Form (CNF).
    
    CNF form is And's of Or's, such as
        And(
            Or(x,y),
            z,
            Or(y,w)
        )
    
    If `names_only`, returns a list of the atom names in each conjunction:
        [
            ["x","y"],
            ["z"],
            ["y","w"]
        ]
    
    """
    if names_only:
        return make_dnf(switch_and_or(expr), names_only=True)
    else:
        return switch_and_or(make_dnf(switch_and_or(expr)))


# Helper functions for parsing expressions.

# PyParsing returns atoms as str objects; we need to convert these to Atoms.
def atomize(arg):
    if isinstance(arg, str):
        return Atom(arg)
    else:
        return arg

make_and = lambda args: And([atomize(x) for x in args[0][0::2]])
make_or = lambda args: Or([atomize(x) for x in args[0][0::2]])

name = pp.Word(pp.alphanums+'_-.')

bool = pp.infixNotation(name,
    [ 
     (pp.Keyword("or"), 2, pp.opAssoc.LEFT, make_or),
     (pp.Keyword("and"), 2, pp.opAssoc.LEFT, make_and)
    ])


def parse_boolean_string(s):
    """Parse a string into an Boolean Expression object."""
    if not s:
        return Empty()
    else:
        parsed = bool.parseString(s)[0]
        return atomize(parsed)
        

if __name__ == '__main__':  
    print(parse_boolean_string('a'))
    print(parse_boolean_string(''))
    print(parse_boolean_string('a and b'))
    print(parse_boolean_string('a and b or c'))
    print(parse_boolean_string('(a and b) or c'))
    print(parse_boolean_string('a and (b or c)').to_rule())