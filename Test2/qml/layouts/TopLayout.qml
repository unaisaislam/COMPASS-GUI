import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "../widgets"

Item {
    height: parent.height
    width: parent.width

    ColumnLayout {
        id: loginControlLayout
        spacing: 10
        anchors.centerIn: parent
        visible: true

        Label {
            id: lblName
            Layout.preferredWidth: 100
            text: "Add Image"
            font.family: "Spotify Mix"
            font.pointSize: 12
            color: "#ffffff"
        }


        /*TextField {
                        id: txtName
                        Layout.preferredWidth: 100
                        text: ""
                    }*/
        Button {
            id: btnOK
            text: "Start!"
            font.family: "Spotify Mix"
            onClicked: {


                /*lblName.text = "Welcome " + txtName.text;
                            var response = mainController.process_name(txtName.text);
                            lblProgress.text = response;
                            console.log(response);*/
                mainController.process_image()
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
