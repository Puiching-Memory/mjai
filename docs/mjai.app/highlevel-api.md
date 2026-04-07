# Bot API

Source: https://mjai.app/docs/highlevel-api

By using the Bot API, you can easily build your own bot without handling the communication protocol directly.

## Setup

Install the mjai package. This package is already installed in the runtime environment.

```bash
% docker pull docker.io/smly/mjai-client:v3
% docker run --rm -it docker.io/smly/mjai-client:v3 /bin/bash
>>> import mjai
>>> mjai.Bot(player_id=0)
```

You can also use the mjai package in your local environment:
```bash
% pip install mjai
```

## Example code

The rulebase bot is implemented using the Bot API. You can create your own Mahjong AI by overriding the `think()` method of the Bot class. The Bot class has many properties and methods that can be used to define your strategy.

- [Example code](https://github.com/smly/mjai.app/blob/main/examples/rulebase/bot.py)

## Quick check

`bot.py` reads JSON-formatted data from stdin and answers JSON-formatted data to stdout. You can check the response of `bot.py` by giving JSON-formatted data by redirection:

```bash
% python examples/rulebase/bot.py 1 < tests/mjai/bot/data_base_akadora.log
```

## Hand representations

### 1. Short: Riichi-tools-rs hand representation

Defined in [riichi-tools-rs](https://github.com/harphield/riichi-tools-rs). Almost same as the [representation of Tenhou](https://tenhou.net/2), with additional representation for open shapes. See [riichi-tools-rs README](https://github.com/harphield/riichi-tools-rs#hand-representation-parsing).

Example: `"13407m5p335779s46z"`, `"456s11133z(456m1)(456p0)"`

Stored in `bot.tehai` property.

### 2. Vec34: tile count representation

Hand representation with the number of tiles (34-element array).

Example: `[1,0,1,1,1,0,1,0,0,0,0,0,0,1,0,0,0,0,0,0,2,0,1,0,2,0,1,0,0,0,1,0,1,0]`

Stored in `bot.tehai_vec34` property. Akadora and open shapes are not included. Check akadora count with `bot.akas_in_hand`.

### 3. Mjai representation

List of tile strings. Akadora can be represented, but open shapes cannot.

Example: `["1m","3m","4m","5mr","7m","3s","3s","5s","7s","7s","9s","N","F"]`

Stored in `bot.tehai_mjai` property. Check open shapes with `bot.get_call_events()`.

## Properties

### Player State Properties

| Property | Description |
|---|---|
| `last_self_tsumo` | Last tile that the player drew by itself |
| `last_kawa_tile` | Last discarded tile in the game |
| `self_riichi_declared` | Whether the player declared riichi |
| `self_riichi_accepted` | Whether the player accepted riichi |
| `tehai` | Short format of the player's hand |
| `tehai_vec34` | Vec34 format of the player's hand |
| `tehai_mjai` | Mjai format of the player's hand |
| `akas_in_hand` | Number of akadora in the player's hand |
| `shanten` | Number of shanten |

### Action State Properties

| Property | Description |
|---|---|
| `can_discard` | Whether the player can discard a tile |
| `can_act` | Whether the player can act |
| `can_riichi` | Whether the player can declare riichi |
| `can_agari` | Whether the player can call agari |
| `can_tsumo_agari` | Whether the player can call tsumo agari |
| `can_ron_agari` | Whether the player can call ron agari |
| `can_pon` | Whether the player can call pon |
| `can_chi` | Whether the player can call chi |
| `can_chi_low` | Whether the player can call chi with a low tile |
| `can_chi_mid` | Whether the player can call chi with a middle tile |
| `can_chi_high` | Whether the player can call chi with a high tile |
| `can_kan` | Whether the player can call kan |
| `can_ankan` | Whether the player can call ankan |
| `can_daiminkan` | Whether the player can call daiminkan |
| `can_kakan` | Whether the player can call kakan |
| `can_ryukyoku` | Whether the player can abort the round |
| `target_actor` | The actor that the player is targeting for an action |
| `target_actor_rel` | The relative position of the target actor |

### Table State Properties

| Property | Description |
|---|---|
| `kyoku` | The current kyoku (局) |
| `honba` | The current honba (本場) |
| `kyotaku` | The current kyotaku (供託) |
| `jikaze` | The current jikaze (自風) |
| `bakaze` | The current bakaze (場風) |
| `dora_markers` | The current dora markers |
| `scores` | The current scores (ordered by player id) |
| `tiles_seen` | Observed number of tiles from the player |

All properties/methods: [Bot implementation](https://github.com/smly/mjai.app/blob/main/python/mjai/bot/base.py)

## Actions

| Method | Description |
|---|---|
| `action_discard()` | Discard a tile from the player's hand |
| `action_riichi()` | Declare riichi |
| `action_nothing()` | Do nothing (pass to call or declare anything) |
| `action_tsumo_agari()` | Call tsumo agari |
| `action_ron_agari()` | Call ron agari |
| `action_pon()` | Call pon |
| `action_chi()` | Call chi |
| `action_ryukyoku()` | Abort the round |

## Local Development

The mjai package includes a method for simulating a hanchan (半荘) game by specifying a player in a ZIP archive file. See the README of the [mjai.app repository](https://github.com/smly/mjai.app/).
