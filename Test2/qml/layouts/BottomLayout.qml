import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Item {
    height: parent.height
    width: parent.width

    ColumnLayout {
        id: progressLayout
        anchors.centerIn: parent
        spacing: 10

        Label {
            id: lblError
            text: "No image to display!"
            font.family: "Sour Gummy"
            color: "#ffffff"
            visible: true
        }

        Rectangle {
            id: imgContainer
            width: 650
            height: 650
            color: "transparent"
            visible: false

            Image {
                id: imgView
                anchors.centerIn: parent
                source: ""
                // Scale to fit while keeping aspect ratio
                fillMode: Image.PreserveAspectFit
                // Prevent overflow
                width: parent.width
                height: parent.height
                clip: true
            }
        }

        Label {
            id: lblProgress
            Layout.preferredWidth: 100
            text: "v1.4.0"
            font.family: "Sour Gummy"
            color: "#ffffff"
        }
    }

    Connections {
        target: mainController

        function onUpdateProgress(val, msg) {
            lblProgress.text = val + "%: " + msg
            console.log(val + "%: " + msg)
        }

        function onImageChangedSignal(show) {
            if (show) {
                lblError.visible = false
                imgContainer.visible = true
                imgView.source = mainController.get_pixmap()
            } else {
                lblError.visible = true
                imgContainer.visible = false
            }
        }
    }
}
