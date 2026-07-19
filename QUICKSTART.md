# Quick Start: uruchamianie i harmonogram

Praktyczna ściągawka do codziennej obsługi. Pełny opis architektury i wszystkich
komend jest w [README.md](README.md); tu tylko to, co potrzebne żeby odpalić
system i wiedzieć, co się z nim dzieje.

## Uruchomienie harmonogramu

```powershell
cd C:\Users\Dell\Desktop\Workplace\Repositories\Facial-Emotion-Recognition\face-identity-clusterer
.\venv\Scripts\Activate.ps1
python main.py schedule
```

Jedna komenda uruchamia całość — nie trzeba nic odpalać osobno. Scheduler sam
sprawdza godzinę i przełącza się między dwoma trybami:

- **okno zbierania** (`capture_start`–`capture_end`) → nagrywa sesje z kamery
- **poza oknem** → gdy okno się zamknie, sam uruchamia analizę (detekcja,
  klasteryzacja, aktualizacja prototypów, opcjonalny trening klasyfikatora)

To proces **pierwszoplanowy** — blokuje terminal, dopóki nie zatrzymasz go
Ctrl+C albo nie puścisz w tle (patrz niżej).

## Gdzie zmienić godziny

[config.yaml](config.yaml), sekcja `schedule`:

```yaml
schedule:
  capture_start: "07:00"
  capture_end: "15:00"
  camera: "0"                # indeks kamery USB lub adres RTSP/HTTP
  session_minutes: 60        # jedna sesja zbierania na tyle minut
  analysis_enabled: true
  train_after_analysis: true
```

Obsługuje też okno przechodzące przez północ (`capture_start: "22:00"`,
`capture_end: "06:00"`).

Jednorazowe nadpisanie bez edycji pliku:

```powershell
python main.py schedule --capture-start 08:00 --capture-end 20:00
```

## Sprawdzenie stanu bez uruchamiania

```powershell
python main.py schedule --status
```

Pokazuje: w której jesteś fazie teraz, kiedy następne przejście, czy analiza
jest zaległa, kiedy była ostatnia.

## Uruchomienie w tle

```powershell
$venvPython = ".\venv\Scripts\python.exe"
Start-Process -FilePath $venvPython -ArgumentList "main.py","schedule" `
    -RedirectStandardOutput "logs\schedule.log" `
    -RedirectStandardError "logs\schedule_err.log" `
    -WindowStyle Hidden -PassThru
```

Zwraca obiekt z `.Id` (PID) — zanotuj go, żeby móc zatrzymać proces później.
Logi programu lądują na stderr (to normalne, nie błąd) — czyli w
`schedule_err.log`, nie w `schedule.log`.

Jeśli zgubisz PID (np. po zamknięciu terminala, w którym go uruchomiłeś),
znajdź go po treści komendy zamiast pamiętać liczbę:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Select-Object ProcessId, CommandLine | Format-List
```

Zatrzymanie:

```powershell
Stop-Process -Id <PID>
```

Sprawdzenie, czy konkretny PID nadal żyje:

```powershell
Get-Process -Id <PID> -ErrorAction SilentlyContinue
```

## Podgląd logów na żywo

| Terminal | Komenda |
|---|---|
| Git Bash / WSL | `tail -f -n 20 logs/schedule_err.log` |
| PowerShell | `Get-Content .\logs\schedule_err.log -Wait -Tail 20` |
| cmd.exe | `powershell -Command "Get-Content .\logs\schedule_err.log -Wait -Tail 20"` |

## Co jeszcze można podglądać w międzyczasie

| Co | Komenda | Co mówi |
|---|---|---|
| Liczba zebranych klatek | `ls dataset\ -Recurse -Filter *.jpg \| measure` | ile materiału już jest |
| Lista sesji + ich manifesty | `ls dataset\` → `cat dataset\<sesja>\manifest.json` | ile klatek odrzucił każdy filtr (interwał/ostrość/zmiana) |
| Stan schedulera między restartami | `cat dataset\schedule_state.json` | kiedy była ostatnia udana analiza, jej podsumowanie, ostatni błąd |
| Wynik ostatniej analizy | `cat output\summary.json` | liczba osób, twarzy, szumu |
| Wizualna kontrola | `ls output\montages\` | kolaż twarzy per osoba — najszybszy sanity check |
| Wycinki per osoba | `ls output\persons\` | pojedyncze zdjęcia pogrupowane per `person_XXXX` |

Podgląd z kamery "na żywo" (osobne polecenie, koliduje z działającym
schedulerem, bo jedna kamera = jeden uchwyt):

```powershell
python main.py capture --camera 0 --max-frames 5 --display
```

## Normalne ostrzeżenia, którymi nie trzeba się martwić

Sporadyczny błąd przy otwieraniu kamery między sesjami (`Cannot open capture
source: 0`, ostrzeżenia MSMF w logu) jest **oczekiwany i obsłużony** — scheduler
loguje błąd, czeka `retry_seconds` i próbuje ponownie, bez przerywania całego
procesu. Jednorazowy taki wpis przy przejściu między sesjami to normalne
zachowanie sterownika Windows przy zwalnianiu/otwieraniu urządzenia, nie awaria.

Sygnał realnego problemu: ten błąd powtarza się **przy każdej** sesji, nie
sporadycznie.
