import QtQuick
import QtQuick.Controls
import QtQuick.Layouts


Item {
    id: imgFiltersControl
    Layout.preferredHeight: 256
    Layout.preferredWidth: parent.width

    property int cbxWidthSize: 400
    property int valueRole: Qt.UserRole + 4


    ColumnLayout {
        id: imgFiltersCtrlLayout
        spacing: 10

        Repeater {
            id: filterRepeater
            model: imgFilterModel
            delegate: RowLayout {
                Layout.fillWidth: true
                Layout.leftMargin: 10
                Layout.alignment: Qt.AlignLeft

                CheckBox {
                    id: checkBox
                    objectName: model.id
                    Layout.preferredWidth: cbxWidthSize
                    text: model.text
                    font.family: "Spotify Mix"
                    font.pointSize: 13
                    property bool isChecked: model.value
                    checked: isChecked
                    onCheckedChanged: {
                        if (isChecked !== checked) {  // Only update if there is a change
                            isChecked = checked
                            let val = checked ? 1 : 0;
                            var index = imgFilterModel.index(model.index, 0);
                            imgFilterModel.setData(index, val, valueRole);
                            mainController.apply_filter_changes();
                        }
                    }
                }
            }
        }
    }
}
