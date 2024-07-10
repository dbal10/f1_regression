import pandas as pd
from kivy.app import App
from kivy.core.window import Window
from kivy.lang import Builder
from kivy.uix.button import Button
from kivy.uix.dropdown import DropDown
from kivy.uix.floatlayout import FloatLayout
from kivy.uix.image import Image
from kivy.uix.label import Label
import serving

Builder.load_string("""
<ColoredLabel>:
    canvas.before:
        Color: 
            rgba: 90/255, 0, 0, 1
        RoundedRectangle:
            pos: self.pos
            size: self.size
            radius: [10]
""")


class ColoredLabel(Label):
    pass


def get_races_df():
    races_df = pd.read_csv("./f1db_csv/races.csv", delimiter=",")

    return races_df


def get_data_table(dataframe):
    row_data = dataframe.to_records(index=False)
    return row_data


gp_df = []
result = ''
race_button_created = False


def get_year_race(year):
    if year != 'Select year':
        year = int(year)
        races_df = get_races_df()

        global gp_df
        gp_df = races_df.loc[races_df['year'] == year]
        gp_df = gp_df[['raceId', 'name']]


def get_raceId_to_predict(n):
    global gp_df
    result = gp_df.loc[gp_df['name'] == n, 'raceId'].iloc[0]
    return result


def get_result_table_labels(race, layout):
    row_height = 0.035
    race_id = get_raceId_to_predict(race)

    prediction_result_df = serving.test_f1_serving(race_id)
    prediction_result_df.drop('probability', inplace=True, axis=1)
    prediction_result_df.drop('driverId', inplace=True, axis=1)

    row_data = get_data_table(prediction_result_df)
    column_data = ['Predicted position', 'Driver', 'Real position']

    for idx, column in enumerate(column_data):
        column_label = ColoredLabel(size_hint=(None, None),
                                    size=(300, 30),
                                    color='white',
                                    pos_hint={'center_x': 0.3 + 0.2 * idx, 'center_y': 0.75},
                                    text=column)
        layout.layout.add_widget(column_label)

    for idx, row in enumerate(row_data):
        row_label = ColoredLabel(size_hint=(None, None),
                                 size=(300, 25),
                                 color='white',
                                 pos_hint={'center_x': 0.3, 'center_y': 0.705 - idx * row_height},
                                 text=str(row[0]))
        layout.layout.add_widget(row_label)

    for idx, row in enumerate(row_data):
        row_label = ColoredLabel(size_hint=(None, None),
                                 size=(300, 25),
                                 color='white',
                                 pos_hint={'center_x': 0.3 + 0.2, 'center_y': 0.705 - idx * row_height},
                                 text=str(row[2]) + ' ' + str(row[3]))
        layout.layout.add_widget(row_label)

    for idx, row in enumerate(row_data):
        value = int(row[1])
        if row[1] == -1:
            value = 'DNF'
        else:
            value = str(value)
        row_label = ColoredLabel(size_hint=(None, None),
                                 size=(300, 25),
                                 color='white',
                                 pos_hint={'center_x': 0.3 + 0.4, 'center_y': 0.705 - idx * row_height},
                                 text=value)
        layout.layout.add_widget(row_label)


def on_year_btn_release(layout, raceButton, race_dd, year_button, year_dd, ddButton):
    global race_button_created
    if not race_button_created:
        layout.layout.add_widget(raceButton)
        race_button_created = True
    year_dd.select(ddButton.text)
    get_year_race(year_button.text)
    create_gp_dd(layout, race_dd, raceButton)


def create_gp_dd(layout, dd, race_button):
    global gp_df

    race_button.text = 'Select race'
    dd.clear_widgets()

    # race dropdown rows
    for index, row in gp_df.iterrows():
        btn = Button(text=str(row['name']), size_hint_y=None, height=33, background_color='#FF0000')
        btn.bind(on_release=lambda btn: on_race_btn_release(layout, dd, race_button, btn))

        dd.add_widget(btn)


def on_race_btn_release(layout, race_dd, raceButton, btn):
    race_dd.select(btn.text)
    get_result_table_labels(raceButton.text, layout)


class F1GUI(App):
    def build(self):
        self.layout = FloatLayout()
        Window.size = (1422, 800)
        Window.top = 80
        Window.left = 50

        self.layout.add_widget(Image(source='uae-10opa.jpg'))

        # buttons
        year_button = Button(text="Select year",
                             size_hint=(0.25, 0.07),
                             pos_hint={'center_x': 0.25, 'center_y': 0.88},
                             background_color='#FF0000')
        self.layout.add_widget(year_button)

        race_button = Button(text="Select race",
                             size_hint=(0.25, 0.07),
                             pos_hint={'center_x': 0.75, 'center_y': 0.88},
                             background_color='#ff0000')

        # DropDown instances
        year_dd = DropDown(max_height=505)
        race_dd = DropDown(max_height=505)

        year_button.bind(on_release=year_dd.open)
        race_button.bind(on_release=race_dd.open)

        # year dropdown rows
        for index in range(2021, 1990, -1):
            btn = Button(text=str(index), size_hint_y=None, height=33, background_color='#FF0000')
            btn.bind(on_release=lambda btn: on_year_btn_release(self, race_button, race_dd, year_button, year_dd, btn))
            year_dd.add_widget(btn)

        year_dd.bind(on_select=lambda instance, x: setattr(year_button, 'text', x))
        race_dd.bind(on_select=lambda instance, x: setattr(race_button, 'text', x))

        return self.layout


if __name__ == '__main__':
    F1GUI().run()
