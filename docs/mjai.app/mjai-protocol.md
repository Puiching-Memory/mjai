# Mjai Protocol

Source: https://mjai.app/docs/mjai-protocol

Our Mjai protocol is largely based on [Gimite's original Mjai protocol](https://gimite.net/pukiwiki/index.php?Mjai%20%E9%BA%BB%E9%9B%80AI%E5%AF%BE%E6%88%A6%E3%82%B5%E3%83%BC%E3%83%90), with a few minor changes. The majority of the implementation is based on [Mortal's libriichi](https://github.com/Equim-chan/Mortal). Some rules have been added, such as the rule that players pay a penalty when they make a runtime error.

## Overview

First, 4 players listen for connections as TCP servers. Then, the game simulator sends JSON event messages to the players until the game end.

> **TCP Server?**
> In the submission file, it is not necessary to implement a TCP server. It is only required to communicate JSON messages to the standard IO.

## Tile format

In addition to the 34 basic tile types, there are three types of red dora and unseen tile representations.

- Manzu (萬子): "1m", "2m", ..., "9m"
- Pinzu (筒子): "1p", "2p", ..., "9p"
- Souzu (索子): "1s", "2s", ..., "9s"
- Wind (風牌; Kazehai): "E" (東; Ton), "S" (南; Nan), "W" (西; Shaa), "N" (北; Pei)
- Dragon (三元牌; Sangenpai): "P" (白; Haku), "F" (發; Hatsu), "C" (中; Chun)
- Red dora (赤ドラ; Akadora): "5mr", "5pr", "5sr"
- Unseen tile: "?"

## Example JSON events

The player receives a list of JSON event messages up to the next actionable event. `<-` represents a message from the game simulator. `->` represents a message from a player.

```
<- [{"type":"start_game","id":0}]
-> {"type":"none"}

<- [{"type":"start_kyoku","bakaze":"E","dora_marker":"2s",
     "kyoku":1,"honba":0,"kyotaku":0,"oya":0,
     "scores":[25000,25000,25000,25000],
     "tehais":[
       ["9p","7m","9s","9s","8m","5s","2p","W","C","5s","N","5mr","F"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"]
     ]},
    {"type":"tsumo","actor":0,"pai":"6p"}]
-> {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
```

## Flowchart

See: https://mjai.app/flowchart.png

## Events

### Start Game

`id` represents the seat number in the game. 0 = chiicha (起家; first dealer), 1 = shimocha (下家), 2 = toimen (対面), 3 = kamicha (上家).

The player always returns a "none" event.

```
<- [{"type":"start_game","id":0}]
-> {"type":"none"}
```

### Start Kyoku

The hand tiles (tehais) and scores are arranged in the order of the `id` in the `start_game` event. Since `start_kyoku` events are not actionable, you will not receive only a `start_kyoku` event.

```
<- [{"type":"start_kyoku","bakaze":"E","dora_marker":"2s",
     "kyoku":1,"honba":0,"kyotaku":0,"oya":0,
     "scores":[25000,25000,25000,25000],
     "tehais":[
       ["9p","7m","9s","9s","8m","5s","2p","W","C","5s","N","5mr","F"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"]
     ]},
    {"type":"tsumo","actor":0,"pai":"6p"}]
-> {"type":"dahai","actor":0,"pai":"W","tsumogiri":false}
```

### Tsumo & Dahai

The most common event in a single kyoku. If the `actor` matches the player's ID, the `tsumo` event becomes actionable.

```
<- [{"type":"tsumo","actor":1,"pai":"?"},
    {"type":"dahai","actor":1,"pai":"7s","tsumogiri":false},
    {"type":"tsumo","actor":2,"pai":"?"},
    {"type":"dahai","actor":2,"pai":"F","tsumogiri":true},
    {"type":"tsumo","actor":3,"pai":"?"},
    {"type":"dahai","actor":3,"pai":"2m","tsumogiri":false},
    {"type":"tsumo","actor":0,"pai":"3m"}]
-> {"type":"dahai","actor":0,"pai":"3m","tsumogiri":true}
```

### Call: Pon

```
<- [{"type":"dahai","actor":1,"pai":"5sr","tsumogiri":false}]
-> {"type":"pon","actor":0,"target":1,"pai":"5sr","consumed":["5s","5s"]}

<- [{"type":"pon","actor":0,"target":1,"pai":"5sr","consumed":["5s","5s"]}]
-> {"type":"dahai","actor":0,"pai":"9p","tsumogiri":false}
```

### Call: Chi

```
<- [{"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}]
-> {"type":"chi","actor":0,"target":3,"pai":"4p","consumed":["5p","6p"]}

<- [{"type":"chi","actor":0,"target":3,"pai":"4p","consumed":["5p","6p"]}]
-> {"type":"dahai","actor":0,"pai":"9m","tsumogiri":false}
```

### Call: Kakan

In the case of Kakan (加槓), the next dahai event is followed by a dora event.

```
<- [{"type":"tsumo","actor":0,"pai":"6m"}]
-> {"type":"kakan","actor":0,"pai":"6m","consumed":["6m","6m","6m"]}

<- [{"type":"kakan","actor":0,"pai":"6m","consumed":["6m","6m","6m"]},
    {"type":"tsumo","actor":0,"pai":"1p"}]
-> {"type":"dahai","actor":0,"pai":"2p","tsumogiri":false}

<- [{"type":"dahai","actor":0,"pai":"2p","tsumogiri":false},
    {"type":"dora","dora_marker":"3s"}, ...]
```

### Call: Daiminkan

```
<- [{"type":"dahai","actor":2,"pai":"5m","tsumogiri":true}]
-> {"type":"daiminkan","actor":0,"target":2,"pai":"5m","consumed":["5m","5m","5mr"]}

<- [{"type":"daiminkan","actor":0,"target":2,"pai":"5m","consumed":["5m","5m","5mr"]},
    {"type":"tsumo","actor":0,"pai":"1p"}]
-> {"type":"dahai","actor":0,"pai":"2p","tsumogiri":false}

<- [{"type":"dahai","actor":0,"pai":"2p","tsumogiri":false},
    {"type":"dora","dora_marker":"3s"}, ...]
```

### Call: Ankan

```
<- [{"type":"tsumo","actor":0,"pai":"F"}]
-> {"type":"ankan","actor":0,"consumed":["F","F","F","F"]}

<- [{"type":"ankan","actor":0,"consumed":["F","F","F","F"]},
    {"type":"tsumo","actor":0,"pai":"1p"}]
-> {"type":"dahai","actor":0,"pai":"2p","tsumogiri":false}

<- [{"type":"dahai","actor":0,"pai":"2p","tsumogiri":false},
    {"type":"dora","dora_marker":"3s"}, ...]
```

### Reach (Riichi)

```
<- [{"type":"tsumo","actor":0,"pai":"7p"}]
-> {"type":"reach","actor":0}

<- [{"type":"reach","actor":0}]
-> {"type":"dahai","pai":"3p","actor":0,"tsumogiri":false}

<- [{"type":"dahai","pai":"3p","actor":0,"tsumogiri":false},
    {"type":"reach_accepted","actor":0}, ...]
```

### Hora (Ron Agari)

After hora, `end_kyoku` event follows. If the game is not finished, `start_kyoku` follows and the next kyoku starts.

```
<- [{"type":"dahai","actor":3,"pai":"C","tsumogiri":true}]
-> {"type":"hora","actor":1,"target":3,"pai":"C"}

<- [{"type":"end_kyoku"}]
-> {"type":"none"}

<- [{"type":"start_kyoku",...}, ...]
```

### Hora (Tsumo Agari)

```
<- [{"type":"tsumo","actor":3,"pai":"5s"}]
-> {"type":"hora","actor":3,"target":3}

<- [{"type":"end_kyoku"}]
-> {"type":"none"}

<- [{"type":"end_game"}]
-> {"type":"none"}
```

### Ryukyoku (Abortive Draw)

Abortive draws by kyuusyu-kyuuhai (Nine terminals abortion; 九種九牌).

```
<- [{"type":"start_kyoku","bakaze":"E","dora_marker":"2s",
     "kyoku":1,"honba":0,"kyotaku":0,"oya":0,
     "scores":[25000,25000,25000,25000],
     "tehais":[
       ["1m","9m","1p","9p","1s","1s","E","E","S","S","S","P","F"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
       ["?","?","?","?","?","?","?","?","?","?","?","?","?"]
     ]},
    {"type":"tsumo","actor":0,"pai":"C"}]
-> {"type":"ryukyoku"}
```
