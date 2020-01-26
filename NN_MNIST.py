import tensorflow as tf

# 定数定義
L = tf.keras.layers
batch_size = 128
epochs = 100

# モデルを準備
class NN_model(tf.keras.Model):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.activation = L.ReLU()
        self.flatten = L.Flatten()
        self.dense50 = L.Dense(50, activation=self.activation)
        self.dense10 = L.Dense(10, activation='softmax')

    def call(self, inputs, training=False):
        x = self.flatten(inputs)
        x = self.dense50(x)
        x = self.dense10(x)
        return x

# 前処理を準備
def prepare_mnist_features_and_labels(x, y):
  x = tf.cast(x, tf.float32) / 255.0
  y = tf.cast(y, tf.int64)
  return x, y

# mnistをDounload
mnist = tf.keras.datasets.mnist
train_data, test_data = mnist.load_data()
train_data = tf.data.Dataset.from_tensor_slices(train_data)
test_data = tf.data.Dataset.from_tensor_slices(test_data)

# 前処理の実行とバッチサイズの指定
train_data = train_data.map(prepare_mnist_features_and_labels).batch(batch_size)
test_data = test_data.map(prepare_mnist_features_and_labels).batch(batch_size)

# モデルのインスタンス初期化
model = NN_model()

# モデルコンパイル
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=[tf.keras.metrics.SparseCategoricalAccuracy()]
)

# モデルの実行
model.fit(train_data, epochs=epochs, validation_data=test_data)

