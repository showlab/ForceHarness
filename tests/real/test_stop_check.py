"""The stop check after a primitive: a reverse rebound within tolerance passes; other motion is a safety event."""
import pytest
from forceharness.real.stop_validation import classify_stop


@pytest.mark.parametrize('motion,delta,soft,passed,safety',[
    ([0,0,1],[0,0,-.005194],True,True,False),
    ([0,0,1],[0,0,-.006],True,True,False),
    ([0,0,1],[0,0,-.0061],True,False,False),
    ([0,0,1],[0,0,-.008],True,False,False),
    ([0,0,1],[0,0,-.0081],True,False,True),
    ([0,0,-1],[0,0,.0051],True,False,False),
    ([1,0,0],[-.0051,0,0],True,False,False),
    ([0,0,1],[.0051,0,0],True,False,True),
    ([0,0,1],[.0051,0,-.002],True,False,True),
    ([0,0,1],[0,0,.0051],True,False,True),
    ([0,0,1],[0,0,-.005194],False,False,True),
    ([0,0,0],[0,0,-.005194],True,False,True),
    ([0,0,1],[-.002,0,-.005],True,True,False),
])
def test_rebound_and_safety_boundaries(motion,delta,soft,passed,safety):
    result=classify_stop([[0,0,0],delta],motion,normal_soft_stop=soft)
    assert result['rebound_passed']==passed
    assert result['safety_event']==safety


def test_checks_intermediate_excursions_not_just_final_position():
    result=classify_stop([[0,0,-.009],[0,0,0]],[0,0,1],normal_soft_stop=True)
    assert result['safety_event'] and not result['rebound_passed']
