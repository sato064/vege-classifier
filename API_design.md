# 野菜分類API 仕様書

## 改版履歴
|発行日 | 詳細 |
| :-: | :-:|
|2024/01/10 | 初版発行 |

## 野菜分類API

### 概要
野菜分類のためのAPI

### URL
https://diamond.u-gakugei.ac.jp/vege-api/v1/detect_image

### メソッド
- POST

### リクエストボディ
| パラメータ名 | 型 | 内容 |
| :-: | :-: | :-: | 
| base64_image | string | base64でデコードされた画像 |

#### リクエストボディサンプル
`{
  "base64_image" : "/9j/4QAYRXhpZgAASUkqAAgAAAAAAAAAAAAAAP/sABFEdWNreQABAAQAAAA8AAD/4QMdaHR0cDovL25zLmFkb2JlLmNvbS94YXAvMS4wLwA8P3hwYWNrZXQgYmVnaW49Iu+7vyIgaWQ9Ilc1TTBNcENlaGlIenJlU3pOVGN6a2(略)"
}`

### レスポンス

#### 成功時
- ステータスコード : 200

#### レスポンスオブジェクト
| パラメータ名 | 型 | 内容 |
| :-: | :-: | :-: | 
| detected_object | string | 推定される野菜の種類 |

#### 推定可能な野菜の一覧
| label | 野菜の種類 |
| :-: | :-:|
| cabbage | キャベツ |
| carrot | 人参 |
| cucamber | きゅうり |
| daikon | 大根 |
| eggplant | なす |
| green_onion | ねぎ |
| onion | 玉ねぎ|
| potate | じゃがいも |
| satoimo | さといも |
| shiitake | 椎茸 |
| spinach | ほうれん草 |
| sweetpotate | さつまいも |
| tomato | トマト |
| turnip | かぶ |

#### レスポンスサンプル
`{
    "detected_object": "satoimo"
}
`

### 失敗時

失敗時の処理は未実装

