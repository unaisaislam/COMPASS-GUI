import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Item {
    // Fallback to avoid Design Studio preview warning
    height: parent ? parent.height : 640
    width: parent ? parent.width : 360

    ColumnLayout {
        id: progressLayout
        anchors.centerIn: parent
        spacing: 10

        Label {
            id: lblError
            text: "No image to display!"
            color: "#FF0000"
            visible: true
        }

        Rectangle {
            id: imgContainer
            width: 256
            height: 256
            color: "lightgray"
            visible: false

            Image {
                id: imgView
                anchors.centerIn: parent
                source: ""
                // Scale to fit while keeping aspect ratio
                fillMode: Image.PreserveAspectFit
                // Prevent overflow
                width: parent ? parent.width : 256
                height: parent ? parent.height : 256
                clip: true
            }
        }

        Label {
            id: lblProgress
            Layout.preferredWidth: 100
            text: "v1.0.0"
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
            console.log("Showing: "+ show)
        }
    }
}
