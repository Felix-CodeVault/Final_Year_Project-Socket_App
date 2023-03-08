import time

from flask import Flask, render_template, request, session, redirect, url_for
from flask_socketio import join_room, leave_room, send, emit, SocketIO
import random
from string import ascii_uppercase

app = Flask(__name__)
app.config["SECRET_KEY"] = "sdasd"
socketio = SocketIO(app)

url_endpoint = "https://abidlabs-draw.hf.space/+/api/predict/"

rooms = {}


def generate_unique_code(length):
    while True:
        code = ""
        for _ in range(length):
            code += random.choice(ascii_uppercase)

        if code not in rooms:
            break
    return code


@app.route("/game")
def game():
    return render_template("game_room.html")


@app.route("/", methods=["POST", "GET"])
def login():
    session.clear()
    if request.method == "POST":
        name = request.form.get("name")

        if not name:
            return render_template("login.html", error="Enter a Name", name=name)

        session["name"] = name
        print("User Created: ", name)
        return redirect(url_for("home"))

    return render_template("login.html")


@app.route("/home", methods=["POST", "GET"])
def home():
    name = session["name"]
    if request.method == "POST":
        code = request.form.get("code")
        join = request.form.get("join", False)
        create = request.form.get("create", False)

        # for joining a room
        if join != False and not code:
            render_template("home.html", error="Enter room code", code=code, name=name)

        room = code
        if create != False:
            room = generate_unique_code(5)
            rooms[room] = {"members": 0, "messages": []}
        elif code not in rooms:
            return render_template("home.html", error="Room doesn't exist", code=code, name=name)

        if rooms[room]["members"] >= 2:
            return render_template("home.html", error="Room at Max Capacity", code=code, name=name)

        session["room"] = room
        print(room)
        return redirect(url_for("prep_zone"))

    return render_template("home.html", name=name)


@app.route("/prep_zone", methods=["POST", "GET"])
def prep_zone():
    room = session.get("room")
    if room is None or session.get("name") is None or room not in rooms:
        return redirect(url_for("home"))

    if request.method == "POST":
        play = request.form.get("play-btn", False)
        if play != False:
            if rooms[room]["members"] == 2:
                print(rooms[room]["members"])
                return redirect(url_for("room"))
            else:
                return render_template("prep_zone.html", code=room, error="Second Player Needed",
                                       messages=rooms[room]["messages"])
    return render_template("prep_zone.html", code=room, messages=rooms[room]["messages"])


@app.route("/room")
def room():
    room = session.get("room")
    # if room is None or session.get("name") is None or room not in rooms:
    #     return redirect(url_for("home"))
    return render_template("room.html", messages=rooms[room]["messages"])


@socketio.on("message")
def message(data):
    room = session.get("room")
    if room not in rooms:
        return

    content = {
        "name": session.get("name"),
        "message": data["data"]
    }
    send(content, to=room)
    rooms[room]["messages"].append(content)
    print(f'{session.get("name")} sent: {data["data"]}')


@socketio.on("connect")
def connect(auth):
    room = session.get("room")
    name = session.get("name")
    if not room or not name:
        return
    if room not in rooms:
        leave_room(room)
        return

    # join room
    join_room(room)
    send({"name": name, "message": "has entered the room"}, to=room)
    rooms[room]["members"] += 1
    print(f"{name} connected to room {room}")


@socketio.on("disconnect")
def disconnect():
    room = session.get("room")
    name = session.get("name")
    time.sleep(1)
    leave_room(room)

    if room in rooms:
        rooms[room]["members"] -= 1
        time.sleep(5)
        if rooms[room]["members"] <= 0:
            del rooms[room]

    send({"name": name, "message": "has left the room"}, to=room)
    print(f"{name} disconnected from room {room}")


@socketio.on("canvas_data")
def handle_canvas_data(data):
    # TODO: deals with drawing recognition

    # send data to all canvases
    emit("canvas_data", data, broadcast=True, include_self=False)


@socketio.on("canvas_data_player_2")
def handle_canvas_data_player_2(data):
    # TODO: deals with drawing recognition

    # send data to all canvases
    emit("canvas_data_player_2", data, broadcast=True, include_self=False)


if __name__ == "__main__":
    socketio.run(app, debug=True, allow_unsafe_werkzeug=True)
