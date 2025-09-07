import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"
import QtQuick.Dialogs

Item {
    height: parent.height
    width: parent.width

    ColumnLayout {
        id: loginControlLayout
        spacing: 10
        anchors.centerIn: parent
        visible: true

        Text {
            id: txtWelcome
            text: "Welcome!"
            font.family: "Spotify Mix"
            font.pointSize: 70
            color: "#ffffff"
        }

        Label {
            id: lblName
            Layout.preferredWidth: 100
            text: "Add Image"
            font.family: "Spotify Mix"
            font.pointSize: 20
            color: "#ffffff"
        }


        /*TextField {
                        id: txtName
                        Layout.preferredWidth: 100
                        text: ""
                    }*/
        Button {
            id: btnOK
            text: "Upload"
            font.family: "Spotify Mix"
            font.pointSize: 15
             onClicked: fileDialog.open()  // open dialog when button is clicked
            }
        
        FileDialog {
            id: fileDialog
            title: "Select an Image"
            nameFilters: ["Images (*.png *.jpg *.jpeg *.bmp)"]

            onAccepted: {
                console.log("File selected:", fileDialog.currentFile)
                mainController.process_image(fileDialog.currentFile)  // send to Python
            }

            onRejected: {
               console.log("File selection canceled")
            }
        }
    }

    ColumnLayout {
        id: filterControlLayout
        spacing: 10
        anchors.centerIn: parent
        visible: false

        ImageFilterWidget {}
    }

    Connections {
        target: mainController


        function onImageChangedSignal(show) {
            if (show) {
                loginControlLayout.visible = false
                filterControlLayout.visible = true
            } else {
                loginControlLayout.visible = true
                filterControlLayout.visible = false
            }
        }
    }
}
