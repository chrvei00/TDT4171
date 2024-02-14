p_flavor_strawberry = 0.7
p_flavor_anchovy = 0.3
p_roundshape_strawberry = 0.8
p_roundshape_anchovy = 0.1
p_redwrapper_strawberry = 0.8
p_redwrapper_anchovy = 0.1

def p_redwrapper():
    return p_flavor_strawberry * p_redwrapper_strawberry + p_flavor_anchovy * p_redwrapper_anchovy

def p_roundshape():
    return p_flavor_strawberry * p_roundshape_strawberry + p_flavor_anchovy * p_roundshape_anchovy

def p_red_and_round():
    return p_flavor_strawberry * p_redwrapper_strawberry * p_roundshape_strawberry + p_flavor_anchovy * p_redwrapper_anchovy * p_roundshape_anchovy

def p_strawberry_given_redwrapper_roundshape():
    # We want to calculate P(Strawberry | RedWrapper, RoundShape)
    # We can use Bayes' Theorem to calculate this
    # P(Strawberry | RedWrapper, RoundShape) = P(RedWrapper, RoundShape | Strawberry) * P(Strawberry) / P(RedWrapper, RoundShape)
    # P(RedWrapper, RoundShape | Strawberry) = P(RedWrapper | Strawberry) * P(RoundShape | Strawberry)
    return (p_redwrapper_strawberry * p_roundshape_strawberry * p_flavor_strawberry) / p_red_and_round()

def value_of_candybox (value_strawberry, value_anchovy):
    return p_flavor_strawberry * value_strawberry + p_flavor_anchovy * value_anchovy

print()
print(f"P(RedWrapper) = {p_redwrapper()}")
print(f"P(Strawberry | RedWrapper, RoundShape) = {p_strawberry_given_redwrapper_roundshape()}")
print(f"Value of CandyBox (v_strawberry: 20, value_anchovy: 5) = {value_of_candybox(10, 5)}")
print()
