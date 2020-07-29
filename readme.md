# first_try
[Epoch 190/200] [Index 1333/1334] [D_A loss: 0.0028] [D_B loss: 0.0026] [G loss: adv: 6.0738]epoch: 190, time: 266.51361632347107
[Epoch 191/200] [Index 1333/1334] [D_A loss: 0.0023] [D_B loss: 0.0017] [G loss: adv: 4.6126]epoch: 191, time: 266.1018216609955
[Epoch 192/200] [Index 1333/1334] [D_A loss: 0.0022] [D_B loss: 0.0019] [G loss: adv: 4.4488]epoch: 192, time: 266.81950783729553
[Epoch 193/200] [Index 1333/1334] [D_A loss: 0.0089] [D_B loss: 0.0027] [G loss: adv: 3.8272]epoch: 193, time: 267.0068926811218
[Epoch 194/200] [Index 1333/1334] [D_A loss: 0.0020] [D_B loss: 0.0149] [G loss: adv: 3.1699]epoch: 194, time: 266.94545555114746
[Epoch 195/200] [Index 1333/1334] [D_A loss: 0.0028] [D_B loss: 0.0021] [G loss: adv: 3.8537]epoch: 195, time: 267.01295280456543
[Epoch 196/200] [Index 1333/1334] [D_A loss: 0.0018] [D_B loss: 0.0111] [G loss: adv: 5.2168]epoch: 196, time: 267.34136390686035
[Epoch 197/200] [Index 1333/1334] [D_A loss: 0.0024] [D_B loss: 0.0039] [G loss: adv: 4.3061]epoch: 197, time: 266.79361486434937
[Epoch 198/200] [Index 1333/1334] [D_A loss: 0.0027] [D_B loss: 0.0039] [G loss: adv: 3.9932]epoch: 198, time: 266.3845691680908
[Epoch 199/200] [Index 1333/1334] [D_A loss: 0.0027] [D_B loss: 0.0087] [G loss: adv: 5.7014]epoch: 199, time: 266.36993956565857
[Epoch 200/200] [Index 1333/1334] [D_A loss: 0.0024] [D_B loss: 0.0034] [G loss: adv: 4.4973]epoch: 200, time: 267.51696157455444

Generatorのロス収束が全然できていない．Discriminatorとのロスオーダーの調整が必要そう．
また，バッチサイズが1であることも起因しているかも．

# second_try
バッチ数を1->4にして，マルチGPUで学習させてみる
[Epoch 192/200] [Index 333/334] [D_A loss: 0.0026] [D_B loss: 0.0024] [G loss: adv: 5.0181]epoch: 192, time: 129.88639330863953
[Epoch 193/200] [Index 333/334] [D_A loss: 0.0025] [D_B loss: 0.0086] [G loss: adv: 4.4092]epoch: 193, time: 129.6328957080841
[Epoch 194/200] [Index 333/334] [D_A loss: 0.0026] [D_B loss: 0.0025] [G loss: adv: 4.0369]epoch: 194, time: 128.98768186569214
[Epoch 195/200] [Index 333/334] [D_A loss: 0.0091] [D_B loss: 0.0064] [G loss: adv: 6.4249]epoch: 195, time: 129.82478713989258
[Epoch 196/200] [Index 333/334] [D_A loss: 0.0017] [D_B loss: 0.0021] [G loss: adv: 4.3727]epoch: 196, time: 128.90407752990723
[Epoch 197/200] [Index 333/334] [D_A loss: 0.1988] [D_B loss: 0.0038] [G loss: adv: 3.9444]epoch: 197, time: 129.47661542892456
[Epoch 198/200] [Index 333/334] [D_A loss: 0.0037] [D_B loss: 0.0044] [G loss: adv: 4.8443]epoch: 198, time: 129.87572693824768
[Epoch 199/200] [Index 333/334] [D_A loss: 0.0035] [D_B loss: 0.0021] [G loss: adv: 4.3830]epoch: 199, time: 129.3073902130127
[Epoch 200/200] [Index 333/334] [D_A loss: 0.1510] [D_B loss: 0.0025] [G loss: adv: 5.3961]epoch: 200, time: 130.0063705444336

多少は見た目的に良くなっている気がするがGeneratorのロスが大きい

# third_try
驚愕の真実！　前の実験は全てlr=0.0002でやっていたorz(原論文は0.002.桁が違う...)
原論文通り，バッチ１のlr0.002で再度実験する．
[Epoch 195/200] [Index 1333/1334] [D_A loss: 0.0193] [D_B loss: 0.0139] [G loss: adv: 3.5099]epoch: 195, time: 230.13317227363586
[Epoch 196/200] [Index 1333/1334] [D_A loss: 0.0053] [D_B loss: 0.0072] [G loss: adv: 4.5430]epoch: 196, time: 230.5687961578369
[Epoch 197/200] [Index 1333/1334] [D_A loss: 0.0056] [D_B loss: 0.0163] [G loss: adv: 4.7796]epoch: 197, time: 230.15341520309448
[Epoch 198/200] [Index 1333/1334] [D_A loss: 0.0112] [D_B loss: 0.0238] [G loss: adv: 4.0656]epoch: 198, time: 230.69474172592163
[Epoch 199/200] [Index 1333/1334] [D_A loss: 0.0039] [D_B loss: 0.0243] [G loss: adv: 5.7379]epoch: 199, time: 229.9405632019043
[Epoch 200/200] [Index 1333/1334] [D_A loss: 0.0167] [D_B loss: 0.1853] [G loss: adv: 3.1496]epoch: 200, time: 230.68549132347107
Generatorのロスが下がらない
別の原因がありそう

# fourth_try
学習率の話は嘘，0.0002があっていた
tensorboardでロスや学習率等を確認する．
discriminatorの最終層バイアス削除
Generatorの学習時にDiscriminatorの勾配変化をしないようにするのを忘れていたので，set_requires_gradを定義して利用

tensorboardの結果を見てもGeneratorのロスが全然下がっていない．Discriminatorのロスが早々と収束してしまっているのが原因っぽい．

# fifth_try
デバッグ用試行
バイアス初期化設定
generator最終層にInstanceNormが入っていたのを削除．

identity lossの追加