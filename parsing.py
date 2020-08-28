
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
    
    __repr__ = __str__
    
    
class Empty(object):
    def __str__(self):
        return "<empty>"
    
    def atoms(self):
        return []
    
    def to_rule(self, dictname="x"):
        return ''
    
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
        
    __repr__ = __str__
        

class And(Junction):
    def __init__(self, operands):
        super().__init__("and", operands)
        

class Or(Junction):
    def __init__(self, operands):
        super().__init__("or", operands)


def make_sub(arg):
    if isinstance(arg, str):
        return Atom(arg)
    else:
        return arg
    

make_and = lambda args: And(map(make_sub, [args[0][0], args[0][2]]))
make_or = lambda args: Or(map(make_sub, [args[0][0], args[0][2]]))

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