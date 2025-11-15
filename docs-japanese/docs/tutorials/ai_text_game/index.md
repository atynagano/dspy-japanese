# DSPyを用いた創造的なテキストベースAIゲームの構築

本チュートリアルでは、DPSyのモジュール型プログラミング手法を活用したインタラクティブなテキストベースアドベンチャーゲームの作成方法を解説します。AIが物語生成、キャラクターインタラクション、および適応型ゲームプレイを司る動的なゲームシステムを構築します。

## 作成する内容

以下の機能を備えたインテリジェントなテキストベースアドベンチャーゲームを構築します：

- 動的な物語生成と分岐型ストーリー展開
- AI駆動によるキャラクターインタラクションと対話システム
- プレイヤーの選択に応じて変化する適応型ゲームプレイ
- インベントリ管理とキャラクター成長システム
- ゲーム状態の保存/読み込み機能

## セットアップ手順

```bash
pip install dspy rich typer
```

## ステップ1：基本ゲームフレームワーク

```python
import dspy
import json
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import random
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
import typer

# DSPyの設定
lm = dspy.LM(model='openai/gpt-4o-mini')
dspy.configure(lm=lm)

console = Console()

class GameState(Enum):
    MENU = "menu"
    PLAYING = "playing"
    INVENTORY = "inventory"
    CHARACTER = "character"
    GAME_OVER = "game_over"

@dataclass
class Player:
    name: str
    health: int = 100
    level: int = 1
    experience: int = 0
    inventory: list[str] = field(default_factory=list)
    skills: dict[str, int] = field(default_factory=lambda: {
        "strength": 10,
        "intelligence": 10,
        "charisma": 10,
        "stealth": 10
    })
    
    def add_item(self, item: str):
        self.inventory.append(item)
        console.print(f"[green]アイテム {item} をインベントリに追加しました！[/green]")
    
    def remove_item(self, item: str) -> bool:
        if item in self.inventory:
            self.inventory.remove(item)
            return True
        return False
    
    def gain_experience(self, amount: int):
        self.experience += amount
        old_level = self.level
        self.level = 1 + (self.experience // 100)
        if self.level > old_level:
            console.print(f"[bold yellow]レベルアップ！現在レベル {self.level} になりました！[/bold yellow]")

@dataclass
class GameContext:
    current_location: str = "Village Square"
    story_progress: int = 0
    visited_locations: list[str] = field(default_factory=list)
    npcs_met: list[str] = field(default_factory=list)
    completed_quests: list[str] = field(default_factory=list)
    game_flags: dict[str, bool] = field(default_factory=dict)
    
    def add_flag(self, flag: str, value: bool = True):
        self.game_flags[flag] = value
    
    def has_flag(self, flag: str) -> bool:
        return self.game_flags.get(flag, False)

class GameEngine:
    def __init__(self):
        self.player = None
        self.context = GameContext()
        self.state = GameState.MENU
        self.running = True
        
    def save_game(self, filename: str = "savegame.json"):
        """現在のゲーム状態を保存します。"""
        save_data = {
            "player": {
                "name": self.player.name,
                "health": self.player.health,
                "level": self.player.level,
                "experience": self.player.experience,
                "inventory": self.player.inventory,
                "skills":

## ステップ2：AIを活用したストーリー生成

```python
class StoryGenerator(dspy.Signature):
    """現在のゲーム状態に基づいて動的な物語コンテンツを生成する。"""
    location: str = dspy.InputField(desc="現在の位置情報")
    player_info: str = dspy.InputField(desc="プレイヤーの詳細情報およびステータス")
    story_progress: int = dspy.InputField(desc="現在の物語進行レベル")
    recent_actions: str = dspy.InputField(desc="プレイヤーの直近の行動履歴")
    
    scene_description: str = dspy.OutputField(desc="現在シーンの詳細な描写")
    available_actions: list[str] = dspy.OutputField(desc="プレイヤーが取り得る行動の一覧")
    npcs_present: list[str] = dspy.OutputField(desc="当該ロケーションに存在するNPCの一覧")
    items_available: list[str] = dspy.OutputField(desc="取得可能なアイテムまたは操作可能なオブジェクトの一覧")

class DialogueGenerator(dspy.Signature):
    """NPCの対話内容および応答を生成する。"""
    npc_name: str = dspy.InputField(desc="NPCの名称および種別")
    npc_personality: str = dspy.InputField(desc="NPCの性格特性および背景情報")
    player_input: str = dspy.InputField(desc="プレイヤーが発話または実行した内容")
    context: str = dspy.InputField(desc="現在のゲーム状況および履歴情報")
    
    npc_response: str = dspy.OutputField(desc="NPCの対話応答内容")
    mood_change: str = dspy.OutputField(desc="NPCの気分変化の状態（肯定的/否定的/中立）")
    quest_offered: bool = dspy.OutputField(desc="NPCがクエストを提示するかどうか")
    information_revealed: str = dspy.OutputField(desc="共有される重要な情報内容")

class ActionResolver(dspy.Signature):
    """プレイヤーの行動を処理し、その結果を決定する。"""
    action: str = dspy.InputField(desc="プレイヤーが選択した行動内容")
    player_stats: str = dspy.InputField(desc="プレイヤーの現在

## ステップ3：ゲームインターフェースとインタラクション

```python
def display_game_header():
    """ゲームタイトル画面を表示する"""
    header = Text("🏰 MYSTIC REALM ADVENTURE 🏰", style="bold magenta")
    console.print(Panel(header, style="bright_blue"))

def display_player_status(player: Player):
    """プレイヤーステータスパネルを表示する"""
    status = f"""
[bold]名前:[/bold] {player.name}
[bold]レベル:[/bold] {player.level} (経験値: {player.experience})
[bold]体力:[/bold] {player.health}/100
[bold]スキル:[/bold]
  • 筋力: {player.skills['strength']}
  • 知性: {player.skills['intelligence']}
  • 魅力: {player.skills['charisma']}
  • 隠密: {player.skills['stealth']}
[bold]所持品:[/bold] {len(player.inventory)} アイテム
    """
    console.print(Panel(status.strip(), title="プレイヤーステータス", style="green"))

def display_location(context: GameContext, scene: Dict):
    """現在の位置とシーン情報を表示する"""
    location_panel = f"""
[bold yellow]{context.current_location}[/bold yellow]

{scene['description']}
    """
    
    if scene['npcs']:
        location_panel += f"\n\n[bold]現在出現中のNPC:[/bold] {', '.join(scene['npcs'])}"
    
    if scene['items']:
        location_panel += f"\n[bold]視認可能なアイテム:[/bold] {', '.join(scene['items'])}"
    
    console.print(Panel(location_panel.strip(), title="現在位置", style="cyan"))

def display_actions(actions: list[str]):
    """選択可能なアクションを表示する"""
    action_text = "\n".join([f"{i+1}. {action}" for i, action in enumerate(actions)])
    console.print(Panel(action_text, title="選択可能アクション", style="yellow"))

def get_player_choice(max_choices: int) -> int:
    """プレイヤーの選択を取得し、入力を検証する"""
    while True:
        try:
            choice = typer.prompt("アクションを選択してください (番号入力)")
            choice_num = int(choice)
            if 1 <= choice_num <= max_choices:
                return choice_num - 1
            else:
                console.print(f"[red]1から{max_choices}までの数字を入力してください[/red]")
        except ValueError:
            console.print("[red]有効な数字を入力してください[/red]")

def show_inventory(player: Player):
    """プレイヤーの所持品を表示する"""
    if not player.inventory:
        console.print(Panel("所持品は空です.", title="所持品", style="red"))
    else:
        items = "\n".join([f"• {item}" for item in player.inventory])
        console.print(Panel(items, title="所持品", style="green"))

def main_menu():
    """メインメニューを表示し、選択を処理する"""
    console.clear()
    display_game_header()
    
    menu_options = [
        "1. 新規ゲーム開始",
        "2. ゲームロード", 
        "3. 遊び方説明",
        "4. 終了"
    ]
    
    menu_text = "\n".join(menu_options)
    console.print(Panel(menu_text, title="メインメニュー", style="bright_blue"))
    
    choice = typer.prompt("オプションを選択してください")
    return choice

def show_help():
    """ヘルプ情報を表示する"""
    help_text = """
[bold]遊び方:[/bold

## ステップ4: メインゲームループ

```python
def create_new_character():
    """新規プレイヤーキャラクターを作成する"""
    console.clear()
    display_game_header()
    
    name = typer.prompt("キャラクター名を入力してください")
    
    # スキルポイント配分を伴うキャラクター作成処理
    console.print("\n[bold]キャラクター作成[/bold]")
    console.print("スキルポイントが10ポイント残っています。各スキルに自由に振り分けてください。")
    console.print("基本スキル値は各10からスタートします。\n")
    
    skills = {"strength": 10, "intelligence": 10, "charisma": 10, "stealth": 10}
    points_remaining = 10
    
    for skill in skills.keys():
        if points_remaining > 0:
            console.print(f"残りポイント: {points_remaining}")
            while True:
                try:
                    points = int(typer.prompt(f"{skill}に追加するポイント数 (0～{points_remaining})"))
                    if 0 <= points <= points_remaining:
                        skills[skill] += points
                        points_remaining -= points
                        break
                    else:
                        console.print(f"[red]0～{points_remaining}の範囲で数値を入力してください[/red]")
                except ValueError

## ゲームプレイの具体例

ゲームを起動すると、以下のような流れで進行します：

**キャラクター作成：**
```
🏰 神秘の領域アドベンチャー 🏰

キャラクター名を入力してください: アリア

キャラクター作成
スキルポイントが10ポイント残っています。各スキルに自由に振り分けてください。
基本スキル値は初期値10から始まります。

残りポイント: 10
筋力に追加するポイント (0～10): 2
知力に追加するポイント (0～8): 4
魅力に追加するポイント (0～4): 3
隠密行動に追加するポイント (0～1): 1

アリアさん、神秘の領域へようこそ！
```

**動的シーン生成:**
```
┌──────────── 現在位置 ────────────┐
│ 村の広場                                │
│                                        │
│ あなたはウィローブルック村の賑やかな中心│
│ に立っている。古びた石造りの噴水が楽しげ│
│ に水を吹き出し、商人たちが商品を売り歩き│
│ 子供たちが遊んでいる。古オークの木陰には│
│ 謎めいたフードを被った人物が潜んでいる。│
│                                        │
│ 現在登場しているNPC：村の長老、商人    │
│ 確認できるアイテム：奇妙なメダル、薬草  │
└──────────────────────┘

┌────────── 選択可能な行動 ─────────────┐
│ 1. フードを被った人物に近づく            │
│ 2. 村の長老と話す                        │
│ 3. 商人の商品を見て回る                  │
│ 4. 奇妙なメダルを調べる                  │
│ 5. 噴水近くで薬草を採取する              │
│ 6. 森への道へ向かう                      │
└─────────────────────────────────┘
```

**AI生成対話例：**
```
村の長老との対話...

村の長老：「おや、旅の若者よ。お前の周りには朝もやのように
大きな運命が漂っているのを感じる。古の予言によれば、
勇気の証を携えた者が現れると伝えられている。どうだ、
旅の途中で何か...普通ではないことに気づいたことはないか？」

💼 クエストの機会を検知！
ℹ️ 村の長老は、あなたに関わるかもしれない古の予言について知っている
```

## 今後の開発方針

- **戦闘システム**：ターン制バトルを実装し、戦略的な要素を追加
- **魔法システム**：リソース管理を伴う呪文詠唱システムを導入
- **マルチプレイヤー機能**：ネットワーク対応による協力プレイモードを追加
- **クエストシステム**：分岐可能な複数段階構成の複雑なミッションを実装
- **世界構築**：プロシージャル生成技術を活用したロケーションとキャラクターの自動生成
- **音声要素**：効果音とバックグラウンドミュージックを追加

本チュートリアルでは、DSPyのモジュール型アーキテクチャを活用することで、AIが創造的なコンテンツ生成を担当しつつ、ゲームロジックとプレイヤーの操作感を一貫して維持する、複雑でインタラクティブなシステムを構築できる手法を示しています。
