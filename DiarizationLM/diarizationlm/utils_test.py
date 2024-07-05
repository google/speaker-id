"""Test utils."""
import os
import unittest
from diarizationlm import utils
import datasets


class UtilsTest(unittest.TestCase):

  def test_normalize_text(self):
    text = "Hello,  HI, \"how_are_you?\" Good."
    expected = "hello hi howareyou good"
    self.assertEqual(expected, utils.normalize_text(text))

  def test_speakers_transform(self):
    speakers = ["a", "b", "a", "d"]
    expected = ["1", "2", "1", "3"]
    self.assertListEqual(expected, utils.speakers_transform(speakers))

  def test_get_oracle_speakers(self):
    hyp_spk = "1 1 1 1 2 2 2 2"
    hyp_spk_align = "2 2 2 2 1 1 1 1"
    hyp_spk_oracle = utils.get_oracle_speakers(hyp_spk, hyp_spk_align)
    expected = [1, 1, 1, 1, 2, 2, 2, 2]
    self.assertEqual(expected, hyp_spk_oracle)

  def test_transcript_preserving_speaker_transfer(self):
    src_text = "hello good morning hi how are you pretty good"
    src_spk = "1 1 1 2 2 2 2 1 1"
    tgt_text = "hello morning hi hey are you be good"
    tgt_spk = "1 2 2 2 1 1 2 1"
    expected = "1 1 2 2 2 2 1 1"
    transferred_spk = utils.transcript_preserving_speaker_transfer(
        src_text, src_spk, tgt_text, tgt_spk)
    self.assertEqual(expected, transferred_spk)

  def test_ref_to_oracle(self):
    test_data = {
        "hyp_text": "yo hello hi wow great",
        "hyp_spk": "1 2 3 2 1",
        "ref_text": "hello hi hmm wow great",
        "ref_spk": "1 2 2 3 3",
    }
    self.assertEqual("1 2 3 1 1", utils.ref_to_oracle(test_data))

  def test_hyp_to_degraded(self):
    test_data = {
        "hyp_text": "yo hello hi wow great",
        "hyp_spk": "1 2 3 2 1",
        "ref_text": "hello hi hmm wow great",
        "ref_spk": "1 2 2 3 3",
    }
    self.assertEqual("1 2 2 1 3", utils.hyp_to_degraded(test_data))

  def test_create_diarized_text(self):
    word_labels = ["hi", "how", "are", "you", "good"]
    speaker_labels = ["1", "2", "2", "2", "1"]
    po = utils.PromptOptions(speaker_prefix="<spk:", speaker_suffix=">")
    result = utils.create_diarized_text(
        word_labels, speaker_labels, use_new_line=False, po=po
    )
    self.assertEqual("<spk:1> hi <spk:2> how are you <spk:1> good", result)

  def test_extract_text_and_spk(self):
    completions = "hi <spk:2> how are you <spk:4> good <spk:3 hello"
    po = utils.PromptOptions(speaker_prefix="<spk:", speaker_suffix=">")
    text, spk = utils.extract_text_and_spk(completions, po=po)
    self.assertEqual("hi how are you good hello", text)
    self.assertEqual("1 2 2 2 4 3", spk)

  def test_extract_text_and_spk_with_bad_speaker(self):
    completions = "hi <spk:2> how are you <spk:4> good <spk:u> hello"
    po = utils.PromptOptions(speaker_prefix="<spk:", speaker_suffix=">")
    text, spk = utils.extract_text_and_spk(completions, po=po)
    self.assertEqual("hi how are you good hello", text)
    self.assertEqual("1 2 2 2 4 4", spk)

  def test_generate_prompts_1(self):
    utt = {
        "utterance_id": "dummy",
        "hyp_text": "hi how are you good hello",
        "hyp_spk": "1 2 2 2 1 3",
    }
    po = utils.PromptOptions(
        emit_input_length=48,
        prompt_prefix="START ",
        prompt_suffix=" --> ",
        speaker_prefix="<spk:",
        speaker_suffix=">",
    )
    prompts = utils.generate_prompts(utt, po=po)
    expected = [
        "START <spk:1> hi <spk:2> how are --> ",
        "START <spk:2> you --> ",
        "START <spk:1> good <spk:3> hello --> ",
    ]
    self.assertListEqual(expected, prompts)

  def test_generate_prompts_2(self):
    utt = {
        "utterance_id": "dummy",
        "hyp_text": (
            "hi morning hi how are you I am good and you hello everyone how"
            " are you"
        ),
        "hyp_spk": "1 1 2 2 2 2 1 1 1 1 1 3 3 3 3 3",
    }
    po = utils.PromptOptions(
        emit_input_length=36,
        prompt_prefix="",
        prompt_suffix=" -> ",
        speaker_prefix="<s:",
        speaker_suffix=">",
    )
    prompts = utils.generate_prompts(utt, po=po)
    expected = [
        "<s:1> hi morning <s:2> hi how -> ",
        "<s:2> are you <s:1> I am -> ",
        "<s:1> good and you <s:3> hello -> ",
        "<s:3> everyone how are you -> ",
    ]
    self.assertListEqual(expected, prompts)

  def test_generate_prompts_3(self):
    utt = {
        "utterance_id": "en_6047",
        "hyp_text": (
            "Right We cut the Fox network We put it on our table Please put on"
            " the table We cut the Fox network 1 hour condensed version Okay"
            " They have a condensed version They cut out commercials Yeah you"
            " better not unplug that thing You'll be a mega trouble I had The"
            " phone is We left the portable phone upstairs last night So the"
            " battery ran out Okay so I have I have the phone from the living"
            " room in the bedroom with a cord stretch across so I can lay on"
            " the bed of course Yeah So that's why Lee almost tripped over the"
            " phone On recorder here So anyway the backers the packers have the"
            " Packers played yet This weekend don't tell me anything more Okay"
            " I might be able to follow that That game sometimes they replay it"
            " Sometimes they replay the whole thing so you get to watch all"
            " three hours Sometimes Fox replays a 1-hour highlight And so weird"
            " because the first time we watch the 1-hour thing it was like all"
            " right you know so you get all set up and everything like that"
            " Also it's like wait a minute second quarter you All right Yeah"
            " what they do is you know all those long boring plays Yeah you"
            " know like when they go but you know those are kind of fun to see"
            " too but I mean it's better than seeing no game at all but I'd"
            " rather see the whole 3 hours Oh really Well I was there today Oh"
            " that's right It's Sunday night Yeah Oh it's at Lambeau then Yeah"
            " it was that landlord I know I know we won because you said who"
            " would we be today or brat Who would be playing Cincinnati Yeah it"
            " was the second half was good Oh so if you are covered for time"
            " then don't bother watching the first half hour just watch your"
            " second half hour So the second half is exciting huh Yeah it was"
            " good What was an exciting It wasn't an exciting game but was it"
            " the packer Viking No not the pack or Viking I think it was the"
            " pack of beer game Yeah it was a pack of beer game L That was"
            " awesome That was I mean it goes right down to the last two"
            " seconds Lori was Laurie Laurie taped at one night she was"
            " exhausted and fell right asleep and so I told her to put the tape"
            " in and I I couldn't stay up either I was too tired It would have"
            " gone until like 2:00 in the morning Yeah so so I just didn't I"
            " just didn't watch it either so the next night Lori was I think"
            " she fell asleep or something and I watched the whole game and I"
            " could hardly keep myself quiet I know the next day Lori said Well"
            " you know how's the game and I just looked down to the floor and I"
            " said it's good you got to watch it and don't ask me anything else"
            " Then she watch it the next night I had a meeting to go try to get"
            " home till about 1:00 in the morning So I I came home and she was"
            " just finishing watching it How did you keep quiet while you were"
            " watching it I said Oh it was so hard I said did I give it away"
            " when I when I you know when I told you and she said no but you"
            " made me want to watch it I said are you sure I didn't give it"
            " away She said No I can't believe that they goes back to the last"
            " two seconds you know cuz at first you know you figure there's 15"
            " seconds they got I'll go all the way down the field you know this"
            " isn't We close and then they start making it and you go Oh my"
            " gosh I got to get down there We I I got tickets from a parent"
            " that game a parent called me up and gave me tickets today Or was"
            " last week's game Okay and how do you go with this guy or does he"
            " give you the ticket No he gave me the tickets I went with a"
            " teacher friend of mine I am so disappoint said that that I mean"
            " I'm not disappointed at all I can't use that word I take it back"
            " It's so bizarre that you would call me because I just wrote you a"
            " one-page letter because I knew Cuz I knew Dad was going to call"
            " Yeah So I just wrote you this whole one-piece letter Now I feel"
            " like I've already told you everything So anyway here are the"
            " things that I told you the letters Your house No your house is"
            " beautiful Yes we got the pictures the day before we got the film"
            " Oh good We got that We got the video Oh so we So we get the"
            " pictures and my my closest Filipino friend happened to be here"
            " His wife is the one who lived with us the first year we're in"
            " Buchanan and I'm going to these pictures Actually I opened up the"
            " post office This is funny I open them out the post office as in"
            " fact we got the pictures the day we got the film So I'm in the"
            " when you get a package at the post office you have to go to this"
            " different room and then they look for the package for you So I"
            " was in the package I was in the post office room and this other"
            " guy was going to get the package and I get the pictures and the"
            " people are so nosy"
        ),
        "hyp_spk": (
            "1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 2"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1"
            " 1 1 1 1 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1"
            " 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1"
            " 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1"
        ),
    }
    po = utils.PromptOptions(
        emit_input_length=12000,
        prompt_prefix=(
            "Move potentially misplaced words to the right speaker in the"
            " following transcript.\n"
        ),
        prompt_suffix=" --> ",
    )
    prompts = utils.generate_prompts(utt, po=po)
    for prompt in prompts:
      self.assertGreater(len(prompt), 12000 // 3)
      self.assertLess(len(prompt), 12000)
    self.assertEqual(len(prompts), 1)

  def test_generate_prompts_4(self):
    utt = {
        "utterance_id": "fe_03_00112",
        "hyp_text": (
            "yeah i've i don't really feel that informed about it either it"
            " seems one sided to me no i mean i definitely have the opinion"
            " that if anything is done it should be done uh unil- unilaterally"
            " it should be done through the u._n. because yes u._n. is a"
            " recognized international authority and the u._s. technically is"
            " that's the that's the and yeah and why we never honor the power"
            " that they have i don't understand that either yeah well i mean"
            " basically the united states doesn't have to big powerful country"
            " right yes rather than rather than sin- uh singly which makes us"
            " look even worse yeah i mean no- not only does it make us look"
            " worse it makes the process somewhat unaccountable it makes it a a"
            " war an an aggressive act yes yes um that's not being monitored by"
            " by the authority and i guess i don't understand why now after"
            " after everything last year is is this coming i it's like going to"
            " is last year an excuse for this for this do you think so i don't"
            " know it seems to me do you it seems to me that it's pretty clear"
            " it's maybe something that the bush administration wanted to have"
            " on its agenda when it came in but then oh interesting but then"
            " the change of the mood of the country um after the terrorist"
            " attacks last year um made something possible that maybe the rest"
            " country wouldn't have been ready for before the people weren't"
            " ready at that time to send it would've been a different"
            " atmosphere yeah i think so people would've not been how"
            " interesting ready for war or or sort of chomping at the bit to go"
            " kill someone to go have some revenge on someone even if it's not"
            " the attacker right right the mind set would've been different we"
            " wouldn't have been thinking in that direction probably probably"
            " would've been more sort of i don't know it's a scary it's a scary"
            " part of the world that that i don't understand and i wish i did"
            " more um e- you know that i would've i that i would read more and"
            " understand it more i have a son whose really into it and i should"
            " pay attention more to what he says is he sort of is he just a"
            " person who's interested or is he some sort of just very"
            " interested yeah ver- ah he's not a journalist or uh political"
            " scientist you could say no no just a college student who enjoys"
            " watching b._b._c. and all that yeah but i don't have a i don't"
            " really i guess my opinion would be that i don't have enough of a"
            " a background in it and that i um i'm not sure that we're on the"
            " right course i don't know whether that would that sounds right"
            " doesn't it yeah yeah it seems like we're sort of agreeing very"
            " interesting so yea- we're kind of like what they used to call"
            " what the doves instead of the hawks yeah right i'm aging myself"
            " now see that yes exactly dove i'm sorry i said i'm aging myself"
            " see that oh by saying that they used to be calling it doves"
            " instead of no no no those phrases from from the phr- the phrase"
            " doves and hawks oh because it yes associated with vietnam yes"
            " yeah yeah i've heard it in other contexts i've heard it in the"
            " con- i've heard it in the context of present day israel for"
            " instance oh yes and that election is coming up too i wonder how"
            " that's going to work out with all that's going on i don't know"
            " either well i'm not i really not going to stay on too long is"
            " that okay did we discuss enough well i think actually eh i think"
            " we might need to stay on the full ten minutes to make sure we get"
            " paid are you serious uh i mean it's only like i think ten or"
            " maybe twelve minutes that it goes for but have you have you not"
            " done this before no yeah i did a couple i'm not totally sure but"
            " i would think that they wouldn't want you to just give your i"
            " think they would want you to do the whole time since yo- we are"
            " getting paid i hadn't thought of that i should've probably um"
            " maybe had it more so that it would be at home rather than at work"
            " oh did you have oh you just yeah you're worried that someone's"
            " going to walk by and yeah uh yeah it would maybe maybe could we"
            " finish this call up and yeah i don't know but but i guess if they"
            " if you're supposed to i didn't start timing did you did you no i"
            " didn't i i think we've only i i think we do have a little bit"
            " more time to cover yeah yes i bet we do i def- i i i would like"
            " to finish it just so i can get my money okay okay go head uh so"
            " um it seems that we haven't we don't have a whole lot much more"
            " to say about the iraq situation now do we right you were talking"
            " about the press coverage before i know that like you said we get"
            " a pretty one sided picture probably here in the u._s. my"
            " girlfriend in particular read and i don't in the case of germany"
            " she reads the german presses especially right and i know someone"
            " that does that too and go head i want to hear what your slant is"
            " on that oh well just generally that there's the the you know um"
            " schroeder right is his name the the chancellor who did very well"
            " for himself by opposing right um u._s. military action um and"
            " right it helped his popularity in germany so it's clear that um"
            " in that country at least the uh the mood is definitely again"
            " unilateral and aren't they protesting or something too i don't"
            " know but or or bad feelings i guess yeah that that the way th-"
            " there there stance is the opposite right yeah i think in gen- i"
            " think the majority opinion in fact probably in most european"
            " countries uh with a population at least against u._s. unilateral"
            " action i don't know about against u._n. action right that's and i"
            " think that's probably how they the rest ma- maybe the other"
            " european countries feel the same i don't know though isn- isn't"
            " england still with us i believe that england is but i don't know"
            " what the majority opinion of the population is in en- yeah i i i"
            " think i read somewhere that it's also sort of yes much less uh"
            " committed huh be interesting to just be over there and be a"
            " little fly on the wall and see what they think don't you think"
            " uh-huh yeah i mean i i you'd think actually if you dug around on"
            " the internet you could probably find some polling information"
            " from those countries but i'd like to talk to people yeah in"
            " general wouldn't you rather than just yeah cue from the news or"
            " whatever how do they feel you know but i guess being on a college"
            " campus you could do that too because there's so many"
            " international students that have opinions about how things are"
            " that's definitely true i eh i don't know what your experience is"
            " but but yeah i'm on a college campus i right and and that they"
            " would um maybe weigh in differently than what than what america"
            " would think i have one professor in my department who's very very"
            " vocal on the issue and has a big war uh really statement on his"
            " really door and on his web page really so he's someone who's not"
            " afraid to make a big some other people are probably biting there"
            " tongues more if they're here in the u._s. huh i mean it depends"
            " on where you come from too whether you feel comfortable political"
            " opinion yes huh i i but he's he's he's definitely the other way"
            " huh yeah he definitely is ou- outspoken dove yes uh eh you know"
            " it's interesting that um issues now are not so much about young"
            " people having to go off to war to wear a head band with vietnam"
            " you know the protests about um invol- you know um the peace"
            " protests and and and the draft and all of that it's interesting"
            " now that that doesn't hardly ever comes up as an issue as far as"
            " um you who's going to fight is you know so the two first of all"
            " that there's this issue of yeah as you said who's going to fight"
            " it doesn't seem to come up much and also that it doesn't there"
            " does not the there's not yet at any rate in any sort of right"
            " -mong among could be that you know people of the age that that it"
            " would it really impact and really the other ones would have to go"
            " out and do it you know um you don't hear it there not i don't"
            " think i think they feel insulated or apart from it or something"
            " oh that's somebody else some nameless other person that's not"
            " them it doesn't doesn't effect them you know it could also be"
            " that it actually at this point has wide spread support among"
            " young people oh you think i don't know it seems like i mean my"
            " impression is that hm but um wow i know my my daughter has a"
            " boyfriend who's a marine and i i keep asking if he's going to be"
            " going over but so far no he's not so"
        ),
        "hyp_spk": (
            "1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 2 2 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 2 2 2 1 1 1 1 2 2 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2"
            " 2 2 1 1 1 1 1 1 1 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 2 2 2"
            " 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1"
            " 1 2 1 1 1 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1"
            " 1 1 2 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 2 2 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 2 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 1 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1"
            " 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 2 2 2 2 2 2 2 2 2"
            " 2 2 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 2 1 1 1 1 1 1 2 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 2 2 2 2 2 2"
            " 2 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 1"
            " 1 1 1 1 2 2 2 1 1 1 1 1 1 1 1 1 1 1 1 2 1 1 2 2 2 2 2 2 2 2 2 2 2"
            " 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2"
        ),
    }
    po = utils.PromptOptions(
        emit_input_length=12000,
        prompt_prefix=(
            "Move potentially misplaced words to the right speaker in the"
            " following transcript.\n"
        ),
        prompt_suffix=" --> ",
    )
    prompts = utils.generate_prompts(utt, po=po)
    for prompt in prompts:
      self.assertGreater(len(prompt), 12000 // 3)
      self.assertLess(len(prompt), 12000)
    self.assertEqual(len(prompts), 1)

  def test_json_utterance_reader(self):
    json_file = os.path.join(
        "testdata/example_data.json"
    )
    # The length limit for both prompt and target.
    max_len = 150
    po = utils.PromptOptions(
        emit_input_length=max_len,
        emit_target_length=max_len,
        prompt_prefix="",
        prompt_suffix=" --> ",
        completion_suffix=" END",
        speaker_prefix="<s:",
        speaker_suffix=">",
    )
    reader = utils.JsonUtteranceReader(
        json_files=json_file,
        text_field="hyp_text",
        input_speaker_field="hyp_spk",
        target_speaker_field="hyp_spk_oracle",
        po=po,
    )
    keys = []
    for key, prompt, target in reader.generate_data_tuple():
      keys.append(key)

      self.assertTrue(prompt.startswith("<s:"))
      self.assertTrue(prompt.endswith(" --> "))
      self.assertTrue(target.startswith("<s:"))
      self.assertTrue(target.endswith(" END"))

      self.assertGreater(len(prompt), max_len / 3)
      self.assertGreater(len(target), max_len / 3)
      self.assertLessEqual(len(prompt), max_len)
      self.assertLessEqual(len(target), max_len)

    self.assertEqual(len(keys), 195)
    self.assertEqual("en_0638_seg0", keys[0])
    self.assertEqual("en_0638_seg1", keys[1])

  def test_dataset_from_generator(self):
    json_file = os.path.join(
        "testdata/example_data.json"
    )
    # The length limit for both prompt and target.
    max_len = 150
    po = utils.PromptOptions(
        emit_input_length=max_len,
        emit_target_length=max_len,
        prompt_prefix="",
        prompt_suffix=" --> ",
        completion_suffix=" END",
        speaker_prefix="<s:",
        speaker_suffix=">",
    )
    reader = utils.JsonUtteranceReader(
        json_files=json_file,
        text_field="hyp_text",
        input_speaker_field="hyp_spk",
        target_speaker_field="hyp_spk_oracle",
        po=po,
    )
    ds = datasets.Dataset.from_generator(reader.generate_data_dict)
    entry = ds[0]
    self.assertEqual(entry["uttid"], "en_0638_seg0")

  def test_find_utt_dict(self):
    data_dict = {
        "utterances": [
            {
                "utterance_id": "utt1",
                "hyp_text": "how are you",
            },
            {
                "utterance_id": "utt2",
                "hyp_text": "good morning",
            },
        ]
    }
    result = utils.find_utt_dict("utt2", data_dict)
    self.assertEqual("good morning", result["hyp_text"])

  def test_update_hyp_text_in_utt_dict(self):
    utt_dict = {
        "utterance_id": "utt1",
        "hyp_text": "how are you good morning",
        "hyp_spk": "1 1 1 2 2",
    }

    new_hyp_spk = "hello how are you good morning hi"
    updated = utils.update_hyp_text_in_utt_dict(utt_dict, new_hyp_spk)
    self.assertEqual("1 1 1 1 2 2 2", updated["hyp_spk"])

  def test_postprocess_completions_for_utt(self):
    utt_dict = {
        "utterance_id": "utt1",
        "hyp_text": "how are you good morning",
        "hyp_spk": "1 1 2 2 2",
        "completions": [
            "<speaker:1> hello how are you",
            "<speaker:2> good morning hi there",
        ],
    }
    utils.postprocess_completions_for_utt(utt_dict)
    self.assertEqual(
        "hello how are you good morning hi there", utt_dict["llm_text"]
    )
    self.assertEqual("1 1 1 1 2 2 2 2", utt_dict["llm_spk"])
    self.assertEqual("1 1 1 2 2", utt_dict["hyp_spk_llm"])

  def test_postprocess_completions_for_utt_with_suffix1(self):
    utt_dict = {
        "utterance_id": "utt1",
        "hyp_text": "how are you good morning",
        "hyp_spk": "1 1 2 2 2",
        "completions": [
            "<speaker:1> hello how are you END",
            "<speaker:2> good morning hi there END",
        ],
    }
    po = utils.PromptOptions(completion_suffix=" END")
    utils.postprocess_completions_for_utt(utt_dict, po=po)
    self.assertEqual(
        "hello how are you good morning hi there", utt_dict["llm_text"]
    )
    self.assertEqual("1 1 1 1 2 2 2 2", utt_dict["llm_spk"])
    self.assertEqual("1 1 1 2 2", utt_dict["hyp_spk_llm"])

  def test_postprocess_completions_for_utt_with_suffix2(self):
    utt_dict = {
        "utterance_id": "utt1",
        "hyp_text": "how are you good morning",
        "hyp_spk": "1 1 2 2 2",
        "completions": [
            "<speaker:1> hello how are you END\t\t\t\t",
            "<speaker:2> good morning hi there END 4 ds 32 df2 fd END sdf ",
        ],
    }
    po = utils.PromptOptions(completion_suffix=" END")
    utils.postprocess_completions_for_utt(utt_dict, po=po)
    self.assertEqual(
        "hello how are you good morning hi there", utt_dict["llm_text"]
    )
    self.assertEqual("1 1 1 1 2 2 2 2", utt_dict["llm_spk"])
    self.assertEqual("1 1 1 2 2", utt_dict["hyp_spk_llm"])

  def test_truncate_suffix_and_tailing_text(self):
    self.assertEqual(
        "hello how are you [eod]",
        utils.truncate_suffix_and_tailing_text(
            text="hello how are you [eod]", suffix=""
        ),
    )
    self.assertEqual(
        "hello how are you",
        utils.truncate_suffix_and_tailing_text(
            text="hello how are you", suffix="[eod]"
        ),
    )
    self.assertEqual(
        "hello how are you",
        utils.truncate_suffix_and_tailing_text(
            text="hello how are you [eod] \t\t\t\t fafsdfdsafsd",
            suffix=" [eod]",
        ),
    )

  def test_transfer_llm_completion(self):
    llm_completion = (
        "19:00 <speaker:1> Hello, how are you doing today? <speaker:2> I am"
        " doing well. What about you? <speaker:1> i'm doing well, too. Thank"
        " you. <speaker:2> my name"
    )
    hyp = (
        "<speaker:1> Hello, how are you doing <speaker:2> today? I am doing"
        " well. What about <speaker:1> you? I'm doing well, too. Thank you."
    )
    transferred = utils.transfer_llm_completion(llm_completion, hyp)
    self.assertEqual(
        "<speaker:1> Hello, how are you doing today? <speaker:2> I am doing"
        " well. What about you? <speaker:1> I'm doing well, too. Thank you.",
        transferred,
    )


if __name__ == "__main__":
  unittest.main()
