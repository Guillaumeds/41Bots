# Hand-made transcripts (checkpoint M0)

These 33 files in `crates/engine/tests/transcripts/` are deals written by hand from `docs/RULES.md`, before looking at either program. Each one sets up a situation from the rules, plays it card by card, and states what must happen. Both the engine and the independent reference implementation [a second, deliberately simple copy of the rules, written separately so the two can check each other] will be run against them.

**Your approval of these transcripts is part of the M0 gate** (STATUS task 9, PLAN §13). Both programs are written from the same rules text, so if we both misread a rule, the 10-million-game comparison between them cannot notice. These transcripts are where you can catch that. Where a transcript and a program disagree, the reviewer decides against RULES.md, and the transcript may turn out to be the one that is wrong.

## How to read a file

- **Seats** 0 to 7 sit clockwise and play passes to the left (seat 3 plays after seat 2). Partners sit opposite each other: **team 0** is seats 0 and 4, **team 1** seats 1 and 5, **team 2** seats 2 and 6, and **team 3** seats 3 and 7. Scores are always listed as teams 0, 1, 2, 3.
- **Cards** are written as the rank followed by the suit letter: `5H` is the 5 of hearts, `10S` the 10 of spades, `AD` the ace of diamonds and `JK` the Joker.
- `title:` and `story:` lines explain the deal in words. Start with those.
- `dealer:` gives the first dealer, `hand 3:` the five cards dealt to seat 3, and `turned:` the turned card. `second:` is the next card turned after a Joker, and `undealt:` lists the rest of the pack. `then hand …` lines give the deal for the next hand of the same match.
- `> Play 9S` and `> Burn 2C` are the moves, made by whoever's turn it is. Robbing is never a move: the program hands the turned card over by itself and only asks which card to burn.
- `=` lines are announcements that must appear after the move, for example `= TrickWon trick=1 seat=3 team=3 points=5`, or `= ProvedNoAce seat=4` (seat 4 has shown it does not hold the ace of trumps). `scores:` gives the scores after the move. `legal:` lists exactly the cards the next player may choose from. `result:` states who won the game or the match.
- Lines starting with `#` are notes. Tricks that do not matter to the point being tested are played with any legal cards and carry no checks.

## The transcripts

| File | What it shows |
|---|---|
| `01-five-of-trumps-bonus-ends-game-mid-hand.txt` | Team 2 is on 31 and wins trick 2 with the 5 of trumps. The trick plus its second bonus (+5) make 41, so the game ends at once, mid-hand. |
| `02-forty-one-by-end-of-hand-bonus.txt` | Team 0 is on 35 after trick 5 and reaches exactly 41 only when the +6 end-of-hand bonus for its J of trumps is added. |
| `03-second-bonus-worth-ten-first-still-eleven.txt` | Over three hands, team 1's first bonus is +6 and its second +5. Team 3's first bonus, which comes after team 1's two, is still +6. |
| `04-bonus-flag-resets-in-new-game.txt` | Team 3 takes its first bonus and wins game 1 by a jink. In game 2 its next bonus is +6 again. |
| `05-lowly-trumps-take-the-end-bonus.txt` | Only two trumps reach the players. The end bonus goes to the A♥ (spades trumps), and in the next hand to the 2 of clubs over the 10 of clubs. |
| `06-turned-five-of-trumps-robbed.txt` | The ace holder robs the turned 5 of diamonds at its turn, burns a card (any card may go, aces included) and plays the 5, whose bonus counts at once. |
| `07-turned-five-of-trumps-not-robbed.txt` | The ace is undealt, so the turned 5 stays on the table all hand. The J's bonus comes only at the end of the hand. |
| `08-turned-ace-of-hearts.txt` | A turned A♥ makes hearts trumps and the dealer robs at once. In trick 1 the Joker beats the A♥, which beats the king. |
| `09-turned-plain-ace.txt` | The ace of clubs is turned and the dealer robs at once; any card may be burned, its ace of diamonds and the turned ace included. The A♥ beats the ace of clubs. |
| `10-dealer-holds-ace-robs-as-eighth-card.txt` | The dealer holds the ace and the turned card is ordinary. The other seven play first, then the dealer robs and burns as the eighth card. |
| `11-ace-holder-leads-and-robs-first.txt` | The first player holds the ace, so it robs and burns before leading. |
| `12-ace-of-hearts-holder-does-not-rob.txt` | Clubs are trumps, so the A♥ holder is not asked to rob. The ace of clubs holder robs at its turn. |
| `13-an-ace-may-be-burned.txt` | A robber holding all four aces may burn any of its six cards (H3 as amended 2026-10-14); it burns the ace of clubs. |
| `14-dealer-burns-the-joker.txt` | The dealer takes a turned Joker and burns the Joker itself. Then the A♥ holder robs the next turned card before leading. |
| `15-joker-then-ace-two-dealer-burns.txt` | The Joker is turned, then the ace of spades. The dealer takes both and burns twice, and 11 cards stay undealt. |
| `16-joker-then-dealer-holds-new-ace.txt` | The Joker is turned, then the 9 of clubs, and the dealer holds the ace of clubs. The dealer robs the 9 at its own turn, as the eighth card of trick 1 (H4 as amended 2026-10-14), and burns again. |
| `17-joker-then-ordinary-card-robbed-by-another-seat.txt` | The Joker is turned, then an ordinary card that seat 3 robs at its turn: two burns by two players. |
| `18-joker-then-ordinary-card-ace-undealt.txt` | The Joker is turned, then an ordinary card, with the ace undealt. Not robbing at once proves nothing about the dealer; every player, the dealer last, shows it lacks the ace as it plays to trick 1. |
| `19-forcing-five-led-and-jack-led.txt` | Rule 17. The J led forces out the Joker, the A♥ and every lower trump, but the 5 may be held back. The 5 led forces out every trump. |
| `20-forcing-joker-led-and-ace-of-hearts-led.txt` | Rule 17. The Joker led forces out the A♥ and every lower trump. The A♥ led forces out the ace of trumps and every lower trump, while the 5, J and Joker may be held back. |
| `21-rule-18-both-branches.txt` | If every trump you hold may be held back, you may play any card. If even one may not, you must play a trump. |
| `22-plain-hearts-led-ace-of-hearts-is-not-a-heart.txt` | A plain heart is led. The A♥ holder, with no other heart, may play anything, and the A♥ wins as a trump because no higher trump (5, J or Joker of trumps) is played to that trick. |
| `23-ace-of-hearts-led-when-hearts-are-not-trumps.txt` | The A♥ led is a trump lead. Nobody has to follow with a heart, and a player holding a low trump must play it, not a heart. |
| `24-off-suit-never-wins.txt` | Diamonds (a plain suit) are led and nobody plays a trump: the 3 of diamonds wins against kings of other suits (which do not follow) and against the 2 and ace of diamonds. Any trump played would have won. |
| `25-card-orders-black-two-red-ten-plain-aces.txt` | Among black trumps the 2 beats the 10, and among red trumps the 10 beats the 2. A plain red ace is the lowest card of its suit, and a plain black ace is fourth. |
| `26-jink.txt` | Team 1 takes all five tricks: a jink. It counts as 41 and wins the game at once, whatever the points; no end bonus is needed (the game is reported with reason `jink`). |
| `27-jink-and-41-in-one-hand-count-as-one-game.txt` | A jink that would also have reached 41 with the end bonus counts as one game, and the match goes on. |
| `28-two-teams-reach-41-after-trick-5.txt` | The trick-5 winner reaches 41 and wins the game. Another team's bonus, which would also have made 41, is lost. |
| `29-game-ends-at-trick-3-pending-bonus-lost.txt` | A team reaches 41 at trick 3. Another team's end-of-hand bonus, still waiting for the end of the hand, is lost. |
| `30-final-scores-45-and-46.txt` | A team on 40 takes a trick (45). In the next game, a team on 36 takes the 5-of-trumps trick (46). |
| `32-dealer-moves-left-after-game-ends-mid-hand.txt` | The deal always moves one seat to the left of the previous dealer after every hand, wherever that hand ended (here a game ended mid-hand), and it carries on across games within the same match; only the first dealer of each match is drawn at random (H5). |
| `33-five-game-match-with-money.txt` | A five-game match: each team wins one game, then team 0 wins the decider. Money at a stake of 5: team 0 +30, each other team −10. |
| `34-match-with-stake-two.txt` | A match for a stake of 2: the winners +12, each losing team −4. |

## Points that rest on our reading of the rules (please check these first)

- **30.** When the 5-of-trumps trick takes a team past 41, its bonus still counts, so the final score is 46, not 41.
- **27.** The jink is checked before the end-of-hand bonus, so a jink plus a bonus that would reach 41 is one game won, not two.
- **28.** After trick 5 the trick's points are counted first. The trick winner takes the game, and a bonus still waiting for another team is lost.
- **29.** The same applies to a game that ends early: a bonus still waiting for the end of the hand is lost.
- **05.** When hearts are not trumps, at least two trumps always reach the players, because only 13 cards are out of play. A hand with no trump at all can never happen at an eight-player table.

## Running them, and reading a hand in plain English

`cargo test -p engine --test transcripts` replays every file against the engine and fails on the first disagreement in any file (file and line named). To see what the engine itself says happened in one deal, in words:

```
TRANSCRIPT=15- cargo test -p engine --test transcripts transcript_render -- --ignored --nocapture
```

(`TRANSCRIPT` picks files by part of the name; `TRANSCRIPT_FULL=1` also shows the dealt and burned cards.) The first run of the harness on 13 October 2026 passed all 34 files, so the engine, the independent reference implementation and these hand-written expectations agree.
