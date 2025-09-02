import QtQuick
import QtQuick.Controls
import QtQuick.Layouts
import "layouts"

ApplicationWindow {
    id: mainWindow
    //width: 1024
   // height: 800
    visible: true
    title: "Spindle.exe"
    visibility: Window.FullScreen

    Image{
        anchors.fill:parent
        source: "bg5.jpg"
    }

    /*Component.onCompleted: {
        console.log("Checking controller:", mainController);
        if (!mainController) {
            console.error("mainController is undefined!");
        }
    }*/
    GridLayout {
        anchors.fill: parent
        rows: 2
        columns: 2

        // First row, first column (spanning 2 columns)
        Rectangle {
            Layout.row: 0
            Layout.column: 0
            Layout.rowSpan: 2
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            width: 200
            height:  956
            color: "transparent"

            TopLayout {}
        }

        // Second row, first column (spanning 2 columns)
        Rectangle {
            Layout.column: 1
            Layout.row: 0
            Layout.rowSpan: 2
            Layout.alignment: Qt.AlignVCenter | Qt.AlignHCenter
            width:  500
            height: 956
            color: "transparent"
            BottomLayout{}
        }
        }
    }
