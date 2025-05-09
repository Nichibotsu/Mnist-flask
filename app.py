from flask import Flask, render_template, request, send_file, jsonify,redirect,url_for
from flask_socketio import SocketIO, emit
import plotly.io as pio
import torch
import torchvision
from torcheval.metrics import MulticlassAccuracy
import numpy as np
import plotly.express as px
from torchvision.datasets import MNIST
import io
from threading import Thread

last_graphs = {"loss": "", "acc": ""}

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")  # wichtig für dev

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MNIST_MLP(torch.nn.Module):
    def __init__(self):
        super(MNIST_MLP, self).__init__()
        self.linear1 = torch.nn.Linear(784, 256)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x

def eval_dataset(model, data_loader, metric_fn, max_batches):
    model.eval()
    for eval_step, data in enumerate(data_loader):
        img_batch, lbl_batch = data
        img_batch = img_batch.to(device)
        lbl_batch = lbl_batch.to(device)
        logits = model(img_batch)
        metric_fn.update(logits, lbl_batch)
        if eval_step == (max_batches - 1):
            break
    acc = float(metric_fn.compute().cpu().numpy())
    metric_fn.reset()
    return acc

def data_set():
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,), (0.5,))
    ])
    trainset = torchvision.datasets.MNIST('./data/', train=True, transform=transform, download=True)
    testset = torchvision.datasets.MNIST('./data/', train=False, transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=32, shuffle=True)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=32, shuffle=False)
    return train_loader, test_loader

def train(lr=0.01, n_epochs=2):
    model = MNIST_MLP().to(device)
    train_loader, test_loader = data_set()
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    metric = MulticlassAccuracy(device=device)

    Loss_arr, Acc_arr = [], []
    step_iter = 0

    for epoch in range(n_epochs):
        model.train()
        for step, data in enumerate(train_loader):
            img_batch, lbl_batch = data
            img_batch, lbl_batch = img_batch.to(device), lbl_batch.to(device)
            optimizer.zero_grad()
            logits = model(img_batch)
            batch_loss = loss_fn(logits, lbl_batch)
            batch_loss.backward()
            optimizer.step()

            if step % 500 == 0:
                with torch.no_grad():
                    preds = torch.argmax(logits, dim=1)
                    acc = torch.mean((preds == lbl_batch).double())
                    Loss_arr.append((step_iter, batch_loss.item()))
                    Acc_arr.append((step_iter, acc.item()))
                    send_log(f"Epoche {epoch+1} | Schritt {step} | Loss: {batch_loss.item():.4f} | Acc: {acc:.4f}")
                    step_iter += 1

        test_acc = eval_dataset(model, test_loader, metric, 300)
        send_log(f"✅ Test Accuracy nach Epoche {epoch+1}: {test_acc:.4f}")

    # Am Ende Diagramm senden
    fig1 = px.line(x=[i[0] for i in Loss_arr], y=[i[1] for i in Loss_arr], labels={"x": "Steps", "y": "Loss"})
    fig2 = px.line(x=[i[0] for i in Acc_arr], y=[i[1] for i in Acc_arr], labels={"x": "Steps", "y": "Accuracy"})

    last_graphs["loss"] = pio.to_html(fig1, full_html=False)
    last_graphs["acc"] = pio.to_html(fig2, full_html=False)

    socketio.emit("training_plot", {
        "graph_html_loss": last_graphs["loss"],
        "graph_html_acc": last_graphs["acc"]
    })

def send_log(msg):
    socketio.emit('training_log', {'message': msg})

@app.route('/')
def index():
    return render_template('index.html',
                           graph_html_loss=last_graphs["loss"],
                           graph_html_acc=last_graphs["acc"])

@app.route('/train', methods=['POST'])
def train_model():
    lr = float(request.form['lr'])
    epochs = int(request.form['epochs'])
    thread = Thread(target=train, args=(lr, epochs))
    thread.start()
    return redirect(url_for('index'))

mnist_data = MNIST('./data/', train=False, download=True)
@app.route('/mnist_image/<int:index>')
def get_mnist_image(index):
    image, _ = mnist_data[index]
    buf = io.BytesIO()
    image.save(buf, format='PNG')
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

if __name__ == "__main__":
    socketio.run(app)