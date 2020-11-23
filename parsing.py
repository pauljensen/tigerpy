
import pyparsing as pp

class Atom(object):
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
    
    
class Empty(object):
    def __str__(self):
        return "<empty>"
    
    def atoms(self):
        return []
    
    def to_rule(self, dictname="x"):
        return ''
    
    def has(self, op):
        return False
    
    __repr__ = __str__
    

class Junction(object):
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
    def __init__(self, operands):
        super().__init__("and", operands)
        

class Or(Junction):
    def __init__(self, operands):
        super().__init__("or", operands)


def show_tree(rule, indent='', indent_step='  '):
    if isinstance(rule, Junction):
        print(indent + "+", rule.operator)
        for op in rule.operands:
            show_tree(op, indent=indent + indent_step)
    else:
        print(indent + '|', rule)
  

def pullup_or(rule):
    if isinstance(rule, Or):
        operands = []
        for op in rule.operands:
            operands += pullup_or(op).operands
        return Or(operands)
    elif isinstance(rule, Atom):
        return Or([rule])
    elif isinstance(rule, Empty):
        return Or([])
    else: # And
        operands = [[x] for x in pullup_or(rule.operands[0]).operands]
        for op in rule.operands[1:]:
            new = pullup_or(op).operands
            operands = [x + [y] for x in operands for y in new]
        return Or([And(x) for x in operands])
    
def switch_and_or(rule):
    if isinstance(rule, Or):
        return And([switch_and_or(x) for x in rule.operands])
    elif isinstance(rule, And):
        return Or([switch_and_or(x) for x in rule.operands])
    else:
        return rule
    
def make_dnf(rule, names_only=False):
    rule = pullup_or(rule)
    if names_only:
        return [x.atoms() for x in rule.operands]
    else:
        return rule

def make_cnf(rule, names_only=False):
    if names_only:
        return make_dnf(switch_and_or(rule), names_only=True)
    else:
        return switch_and_or(make_dnf(switch_and_or(rule)))

def make_sub(arg):
    if isinstance(arg, str):
        return Atom(arg)
    else:
        return arg
    

make_and = lambda args: And([make_sub(x) for x in args[0][0::2]])
make_or = lambda args: Or([make_sub(x) for x in args[0][0::2]])

name = pp.Word(pp.alphanums+'_-.')

bool = pp.infixNotation(name,
    [ 
     (pp.Keyword("or"), 2, pp.opAssoc.LEFT, make_or),
     (pp.Keyword("and"), 2, pp.opAssoc.LEFT, make_and)
    ])


def parse_boolean_string(s):
    if not s:
        return Empty()
    else:
        parsed = bool.parseString(s)[0]
        if isinstance(parsed, str):
            return Atom(parsed)
        else:
            return parsed

if __name__ == '__main__':  
    print(parse_boolean_string('a'))
    print(parse_boolean_string(''))
    print(parse_boolean_string('a and b'))
    print(parse_boolean_string('a and b or c'))
    print(parse_boolean_string('(a and b) or c'))
    print(parse_boolean_string('a and (b or c)').to_rule())