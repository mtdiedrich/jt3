from datetime import date


from jt3.models import Category, Clue, Contestant, Episode, Round


def test_contestant_defaults():
    c = Contestant(name="Alice", description="teacher from Omaha, NE")
    assert c.name == "Alice"
    assert c.description == "teacher from Omaha, NE"
    assert c.player_id is None


def test_contestant_with_player_id():
    c = Contestant(name="Bob", description="engineer", player_id=42)
    assert c.player_id == 42


def test_clue_defaults():
    clue = Clue(
        clue_id="J_1_1",
        order=3,
        value=200,
        is_daily_double=False,
        text="This element has atomic number 1",
    )
    assert clue.clue_id == "J_1_1"
    assert clue.order == 3
    assert clue.value == 200
    assert clue.is_daily_double is False
    assert clue.text == "This element has atomic number 1"
    assert clue.correct_response is None
    assert clue.answerer is None


def test_clue_daily_double():
    clue = Clue(
        clue_id="J_6_1",
        order=6,
        value=1000,
        is_daily_double=True,
        text="The capital of France",
        correct_response="Paris",
        answerer="Carol",
    )
    assert clue.is_daily_double is True
    assert clue.correct_response == "Paris"
    assert clue.answerer == "Carol"


def test_category():
    clues = [
        Clue(clue_id="J_1_1", order=1, value=200, is_daily_double=False, text="Q1"),
        Clue(clue_id="J_1_2", order=5, value=400, is_daily_double=False, text="Q2"),
    ]
    cat = Category(name="SCIENCE", clues=clues)
    assert cat.name == "SCIENCE"
    assert len(cat.clues) == 2
    assert cat.comments is None


def test_category_with_comments():
    cat = Category(name="HISTORY", comments="All from the 20th century", clues=[])
    assert cat.comments == "All from the 20th century"


def test_round():
    r = Round(name="Jeopardy!", categories=[Category(name="CAT1", clues=[])])
    assert r.name == "Jeopardy!"
    assert len(r.categories) == 1


def test_episode():
    ep = Episode(
        game_id=9418,
        show_number=9536,
        air_date=date(2026, 4, 6),
        contestants=[Contestant(name="Alice", description="teacher")],
        rounds=[Round(name="Jeopardy!", categories=[])],
    )
    assert ep.game_id == 9418
    assert ep.show_number == 9536
    assert ep.air_date == date(2026, 4, 6)
    assert len(ep.contestants) == 1
    assert len(ep.rounds) == 1


def test_episode_defaults():
    ep = Episode(game_id=1, show_number=None, air_date=None)
    assert ep.contestants == []
    assert ep.rounds == []
