import QtQuick 2.15
import QtQuick.Controls 2.15

Column {
    id: functionSelector
    spacing: 8
    property alias imagePath: imagePathField.text

    TextField {
        id: imagePathField
        placeholderText: "Enter image path"
        width: parent.width
    }

    CheckBox {
        id: binarizeCheck
        text: "Extract Binary Image"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "binarize")
        }
    }

    CheckBox {
        id: skeletonizeCheck
        text: "Extract Skeleton Graph"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "skeletonize")
        }
    }

    CheckBox {
        id: graphCheck
        text: "Extract Colored Components Graph"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "extractgraph")
        }
    }
    CheckBox {
        id: engraphCheck
        text: "Extract Edge-Node Graph"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "en_graph")
        }
    }

    CheckBox {
        id: deghmCheck
        text: "Extract Degree Heatmap"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "deg_hm")
        } 
    }

    CheckBox {
        id: bchmCheck
        text: "Extract Betweenness-Centrality Heatmap"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "bc_hm")
        }
    }

    CheckBox {
        id: nodesCheck
        text: "Compute Number of Nodes"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "nodes")
        }
    }

    CheckBox {
        id: edgesCheck
        text: "Compute Number of Edges"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "edges")
        }
    }
        CheckBox {
        id: avdegCheck
        text: "Compute Average Degree of Graph"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "avdeg")
        }
    }
        CheckBox {
        id: graphdensCheck
        text: "Compute Density of Graph"
        onClicked: {
            if (checked)
                mainController.apply_functions(imagePathField.text, "graphdens")
        }
    }

}