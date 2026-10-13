# 41: Rules of the game (authoritative)

This is the rules text as supplied by the project owner. The engine must implement it exactly. The house rules in section H decide every point the main text leaves open.

## The table

1. Eight players play in pairs. Partners sit opposite each other.
2. One pack of 53 cards: the 52 ordinary cards and one Joker.
3. The first dealer is chosen at random. After every hand the deal moves one seat to the left.
4. The dealer gives each player two cards and then three, starting with the player on the left. Other bundles are fine, as long as everyone gets five cards in two rounds and nobody gets all five at once.
5. The next card of the pack is turned face up. Its suit is trumps for the hand. The rest of the pack stays face down and out of play.

## Which cards beat which

6. Trumps from best to worst: the 5 of trumps, the J of trumps, the Joker, the ace of hearts, the ace of trumps, king, queen, then the numbers. In a red trump suit the numbers run 10 down to 2. In a black trump suit they run 2 up to 10, so the 2 beats the 10.
7. The Joker and the ace of hearts are trumps in every hand, whatever suit is turned.
8. When hearts are trumps, the ace of hearts is fourth best and there are 14 trumps instead of 15. In any other trump suit, the ace of that suit comes straight after the ace of hearts.
9. In a red suit that is not trumps, the order is king, queen, jack, 10 down to 2, then the ace. The ace is the lowest red card.
10. In a black suit that is not trumps, the order is king, queen, jack, ace, then 2 up to 10. The ace beats the 2, and the 10 is the lowest black card.
11. The ace of hearts never belongs to the plain hearts, because it is always a trump.

## Playing a trick

12. The player left of the dealer leads the first trick. After that, the winner of each trick leads the next.
13. Any card may be led, trumps included.
14. When a plain suit is led, you must play that suit or a trump. You may play a card of another suit only when you have none of the suit led.
15. If you have none of the suit led, you may play any card at all. It can be a trump, but it does not have to be.
16. When a trump is led, you must play a trump if you have one. The exception: the four top trumps, the 5, the J, the Joker and the ace of hearts, may be held back when a lower trump is led. This is reneging.
17. A top trump can be held back only against a lower trump. If a higher trump is led, it must be played like any other trump. So the 5 led forces out every trump. The J led forces out the Joker and the ace of hearts. The Joker led forces out the ace of hearts.
18. If every trump you hold is one you may hold back, you may play any card, even a plain one.
19. The trick goes to the highest trump in it. If no trump is played, it goes to the highest card of the suit led. A card of any other suit never wins.

## Robbing the turned card

20. The player holding the ace of trumps must rob the turned card before playing their first card. To rob it, they burn one card from their hand: they put it face down on the table, where it stays hidden for the whole hand.
21. The ace of trumps itself may not be burned. Nobody sees the burned card, but everyone now knows that player holds the ace of trumps.
22. If the turned card is an ace, the dealer must rob it at once, before the first card is led, and burns a card the same way.
23. If the ace of trumps is in the undealt pack, nobody robs. The turned card stays face up on the table for the whole hand and is not played.
24. If a Joker is turned, the dealer must take it, burning a card; there is no choice, and a dealer who does not take it is turned. The next card of the pack is then turned for trumps, and rules 20 to 23 apply to that card. If that card is an ace of trumps, the dealer must rob it as well and burn another card, else the hand is turned. A dealer who holds the ace of trumps must likewise rob that newly turned card, burning another card. The dealer may not burn the ace of trumps anywhere.
25. A player who holds the ace of trumps and plays a card without robbing has their cards turned face down, and so does their partner. The pair plays no more cards that hand and scores nothing.
26. A dealer who refuses or forgets to rob a turned ace is punished the same way.

## Scoring

27. Each trick is worth 5 points.
28. The trick holding the best trump played in the hand is worth 11 points, 5 for the trick and 6 extra, the first time in a game that a partnership takes it. Every later time in the same game it is worth 10 to that partnership, 5 and 5, while each other partnership still gets 11 for its own first one.
29. If the 5 of trumps is played, its trick is the best-trump trick, and the points count as soon as that trick is won.
30. If the 5 is never played, the bonus goes at the end of the hand to the trick that held the highest trump played, whatever that trump is.
31. A hand in which no trump at all is played has no best-trump trick, and thus no bonus: just five tricks of 5 each.
32. The game is won by the first pair to reach 41. The game ends the moment a pair gets there, even in the middle of a hand, and the rest of the hand is not played.
33. A pair that wins all five tricks of a hand makes a jink and wins the game at once, whatever the score.
34. If every other partnership has been turned, the partnership still in play takes every trick of the hand, the turned partnerships' tricks included under rule 39, and that is a jink: it wins the game.
35. The money is the match: the first pair to win two games.
36. In each new game the scores go back to nought. The games each partnership has won are kept.

## Penalties

37. An illegal card is not always caught when it is played. The partnership's cards are turned the moment a later card proves it: a card of the suit the player claimed not to hold (played off-suit when it was led), a trump the player should have played, or the ace the player failed to rob with.
38. Anyone can break the rules on purpose, for example by taking the turned card without the ace. The partnership's cards are turned as soon as the cards shown at the table prove it: the ace of trumps appearing in another hand, the hand ending without it, or the pair trying to claim the game.
39. Once a partnership is turned, none of its cards counts any more. Every trick it had won in the hand passes, with its points, to the partnership still in play that played the strongest card in that trick. If the highest trump of the hand was one of the turned cards, the bonus goes to the highest trump played in the hand by a partnership that is not turned, and counts for that partnership as rule 28 says.
40. When a partnership is turned in the middle of a trick, the trick goes on among the other players without the turned cards. If the player who was to lead is turned, the lead passes to the next player on the left who is still in.
41. The turned partnership scores nothing for that hand.

---

## H. House rules and decisions (owner-confirmed; these settle open points)

- **H1. Illegal plays are impossible.** The engine offers only legal actions, and all forced actions (robbing) happen automatically. So rules 25, 26, 34 and 37–41 can never trigger and are not implemented as gameplay; tests assert that illegal actions are rejected.
- **H2. Robbing timing (rule 20).** The ace-of-trumps holder robs **at their own first turn in trick 1**, immediately before playing their first card. If they lead trick 1, they rob before leading. They choose the burn after seeing any trick-1 cards already played.
- **H3. Burn choice.** Any card may be burned **except the ace of trumps**, including the card just taken (the turned card, or the Joker for the dealer).
- **H4. Dealer robs at once (rules 22 and 24).** A turned ace is robbed by the dealer at once, before the first lead. With a turned Joker, the dealer takes it and burns at once, then the next card is turned. If that card is an ace, or if the dealer holds the ace of trumps, the dealer robs the new card **at once, always**, burning another card. Otherwise rule 20 / H2 applies to whoever holds the ace of trumps, or rule 23 if it is undealt.
- **H5. Next dealer after a game ends mid-hand:** the deal **always moves one seat left**, exactly as after any hand.
- **H6. The money.** A match is won by the first partnership to win 2 games. Each player of every losing partnership pays the agreed stake to the winning partnership. The pub stake is €5: each winner receives €15 and each loser pays €5. The stake is configurable.
- **H7. What bots play for.** Each partnership maximises its **expected money**, which equals maximising its chance of winning the match. Points, tricks and games have no value of their own; they matter only through their effect on the chance of the money. The chance of the money is re-evaluated after every trick (see docs/PLAN.md §5.3).

## Interpretation notes for implementers

- **Seats and teams.** Seats 0–7 run clockwise, and "left" means the next seat (seat + 1 mod 8). Partners are opposite: team = seat mod 4, so seats 0 & 4, 1 & 5, 2 & 6, 3 & 7 are partners. Play proceeds to the left.
- **The ace of trumps.**
  - It is the ace of the trump suit. When hearts are trumps, it is the A♥.
  - A turned A♥ makes hearts trumps, so it is then the ace of trumps (rule 22 applies).
- **Trump count.**
  - Trump suit is not hearts: 15 trumps, in this order: 5, J, Joker, A♥, A, K, Q, then the numbers (2, 3, 4, 6, 7, 8, 9, 10).
  - Trump suit is hearts: 14 trumps; the A♥ is 4th and there is no separate "ace of trumps" rank.
- **Number order inside trumps (excluding the 5).**
  - Red trump suit: 10, 9, 8, 7, 6, 4, 3, 2.
  - Black trump suit: 2, 3, 4, 6, 7, 8, 9, 10.
- **Reneging test** for a trump led with card L. A trump t in hand may be held back if it is one of the top four (5, J, Joker, A♥) **and** ranks higher than L.
  - If the hand holds at least one trump that may not be held back, the player must play a trump (any trump).
  - Otherwise the player may play any card.
- **Plain-suit lead S.**
  - If the hand holds any card of S, the legal cards are those cards plus every trump.
  - Otherwise any card is legal.
- **Bonus bookkeeping.**
  - Each team has a per-game flag recording whether it has already taken a best-trump bonus in this game. Its first bonus is +6; later bonuses are +5.
  - If the 5 of trumps is played, its trick gets the bonus immediately when won. Otherwise the bonus is assigned after trick 5 to the trick holding the highest trump played. There is no bonus if no trump was played.
- **End checks after trick 5.**
  - Award the trick first, then check for 41.
  - Then check for a jink.
  - Then award the end-of-hand bonus and check for 41 again.
  - A team reaching 41 at any point ends the game immediately.
- **Card counts.**
  - 40 cards are dealt, 1 is turned, and 12 stay undealt.
  - With a turned Joker, 11 stay undealt plus the newly turned card.
  - The dealer may burn up to 2 cards in one hand.
