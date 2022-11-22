__Постановка задачи.__  
Нужно написать приложение, которое будет считывать и выводить кадры с веб- камеры. В процессе считывания определять что перед камерой находится человек, задетектировав его лицо на кадре. После этого, человек показывает жесты руками, а алгоритм должен считать их и определенным образом реагировать на эти жесты. На то, как система будет реагировать на определенные жесты - выбор за вами. Например, на определенный жест (жест пис), система будет здороваться с человеком. На другой, будет делать скриншот экрана. И т.д. Для распознавания жестов, вам надо будет скачать датасет для жестов рук.

__Решение__  
Для решения задачи (после долгих раздумий связанных с выбором датасета) было принято решение самому создать мини-датасет с использованием веб-камеры с которой в дальнейшем и должна будет работать программа. Данные собирались (и размечались) автоматически с использованием - mediapipe.

В данном ноутбуке программа обучается на этих данных и в режиме онлайн выводит название жеста а так же ведет журнал жестов. При появлении жеста "Ok" происходит запись картинки на диск.