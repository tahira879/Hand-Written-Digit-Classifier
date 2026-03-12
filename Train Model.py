"""
train_model.py — Generates a synthetic handwritten-digit dataset from scratch
(no internet needed) and trains a CNN classifier on it.

Run once before the app:
    python train_model.py
"""
import numpy as np, os
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks

np.random.seed(42)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# ── 1. Synthetic digit generator ──────────────────────────────────────────────
def draw_digit(digit, size=28):
    img  = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(img)
    jx, jy = np.random.randint(-3,4), np.random.randint(-3,4)
    sc  = np.random.uniform(0.55, 0.88)
    lw  = np.random.randint(2, 5)
    cx  = size//2 + jx;  cy = size//2 + jy
    r   = int(size * sc * 0.38)

    if digit == 0:
        rx,ry = int(r*np.random.uniform(0.7,1.0)), int(r*np.random.uniform(1.0,1.3))
        draw.ellipse([cx-rx,cy-ry,cx+rx,cy+ry], outline=255, width=lw)
    elif digit == 1:
        draw.line([cx,cy-r,cx,cy+r], fill=255, width=lw)
        if np.random.rand()>0.4: draw.line([cx-r//3,cy-r+4,cx,cy-r], fill=255, width=lw)
        if np.random.rand()>0.5: draw.line([cx-r//2,cy+r,cx+r//2,cy+r], fill=255, width=lw)
    elif digit == 2:
        draw.arc([cx-r,cy-r,cx+r,cy+2], 210,360, fill=255, width=lw)
        draw.arc([cx-r,cy-r,cx+r,cy+2], 0,10, fill=255, width=lw)
        draw.line([cx+r,cy,cx-r+2,cy+r*2-2], fill=255, width=lw)
        draw.line([cx-r,cy+r*2,cx+r,cy+r*2], fill=255, width=lw)
    elif digit == 3:
        draw.arc([cx-r,cy-r,cx+r,cy+2],     300,240, fill=255, width=lw)
        draw.arc([cx-r,cy-2,cx+r,cy+r+r//2],300,240, fill=255, width=lw)
    elif digit == 4:
        draw.line([cx-r+2,cy-r,cx-r+2,cy+2], fill=255, width=lw)
        draw.line([cx-r+2,cy+2,cx+r,cy+2],   fill=255, width=lw)
        draw.line([cx+r//2,cy-r,cx+r//2,cy+r+r//2], fill=255, width=lw)
    elif digit == 5:
        draw.line([cx+r,cy-r,cx-r,cy-r], fill=255, width=lw)
        draw.line([cx-r,cy-r,cx-r,cy],   fill=255, width=lw)
        draw.arc([cx-r,cy-r//3,cx+r,cy+r+r//3], 160,380, fill=255, width=lw)
    elif digit == 6:
        draw.arc([cx-r,cy-r,cx+r,cy+r], 40,360, fill=255, width=lw)
        rx2=int(r*0.65); draw.ellipse([cx-rx2,cy,cx+rx2,cy+r+r//4], outline=255, width=lw)
    elif digit == 7:
        draw.line([cx-r,cy-r,cx+r,cy-r], fill=255, width=lw)
        draw.line([cx+r,cy-r,cx-r//3,cy+r+r//2], fill=255, width=lw)
        if np.random.rand()>0.5: draw.line([cx-r//3,cy+2,cx+r//3,cy+2], fill=255, width=lw)
    elif digit == 8:
        draw.ellipse([cx-r//2,cy-r,cx+r//2,cy], outline=255, width=lw)
        draw.ellipse([cx-r//2-1,cy,cx+r//2+1,cy+r+r//4], outline=255, width=lw)
    elif digit == 9:
        rx2=int(r*0.6); draw.ellipse([cx-rx2,cy-r,cx+rx2,cy], outline=255, width=lw)
        draw.line([cx+rx2,cy-r//3,cx+rx2,cy+r], fill=255, width=lw)

    arr = np.array(img, dtype=np.float32)/255.0
    return np.clip(arr + np.random.normal(0,0.04,arr.shape), 0, 1)

# ── 2. Build / load dataset ──────────────────────────────────────────────────
CACHE = "mnist_data"
N_TR, N_TE = 3000, 500

def build_dataset():
    os.makedirs(CACHE, exist_ok=True)
    tr_x = os.path.join(CACHE,"X_train.npy"); te_x = os.path.join(CACHE,"X_test.npy")
    tr_y = os.path.join(CACHE,"y_train.npy"); te_y = os.path.join(CACHE,"y_test.npy")
    if all(os.path.exists(p) for p in [tr_x,tr_y,te_x,te_y]):
        print("📂 Loading cached dataset …")
        return (np.load(tr_x).reshape(-1,28,28,1), np.load(tr_y),
                np.load(te_x).reshape(-1,28,28,1), np.load(te_y))
    print("🖊️  Generating synthetic dataset …")
    Xtr,ytr,Xte,yte=[],[],[],[]
    for d in range(10):
        print(f"  digit {d}", end=" ")
        for _ in range(N_TR): Xtr.append(draw_digit(d)); ytr.append(d)
        for _ in range(N_TE): Xte.append(draw_digit(d)); yte.append(d)
        print("✓")
    Xtr,ytr=np.array(Xtr,dtype=np.float32),np.array(ytr)
    Xte,yte=np.array(Xte,dtype=np.float32),np.array(yte)
    idx=np.random.permutation(len(Xtr)); Xtr,ytr=Xtr[idx],ytr[idx]
    for arr,path in [(Xtr,tr_x),(ytr,tr_y),(Xte,te_x),(yte,te_y)]: np.save(path,arr)
    print(f"💾 Saved to {CACHE}/")
    return Xtr.reshape(-1,28,28,1), ytr, Xte.reshape(-1,28,28,1), yte

# ── 3. Model ─────────────────────────────────────────────────────────────────
def build_model():
    m = models.Sequential([
        layers.Input(shape=(28,28,1)),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32,(3,3),activation='relu',padding='same'),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64,(3,3),activation='relu',padding='same'),
        layers.MaxPooling2D(), layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(256,activation='relu'),
        layers.BatchNormalization(), layers.Dropout(0.5),
        layers.Dense(10,activation='softmax'),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return m

# ── 4. Train ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("="*55)
    print("  Handwritten Digit Classifier — Training Script")
    print("="*55)
    Xtr, ytr, Xte, yte = build_dataset()
    print(f"  Train {Xtr.shape}  |  Test {Xte.shape}")
    model = build_model()
    model.summary()
    cb = [
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5, verbose=1),
        callbacks.EarlyStopping(patience=6, restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint("mnist_model.h5", save_best_only=True, verbose=1),
    ]
    print("\n🏋️  Training …")
    model.fit(Xtr, ytr, epochs=20, batch_size=256,
              validation_data=(Xte,yte), callbacks=cb, verbose=1)
    loss, acc = model.evaluate(Xte, yte, verbose=0)
    print(f"\n✅  Test Accuracy: {acc*100:.2f}%   Loss: {loss:.4f}")
    print("💾  Saved → mnist_model.h5")
    print("🚀  Launch:  streamlit run app.py")
