import { Injectable } from '@angular/core';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { UserUpdatedPayload } from '../user-updated/user-updated.service';
import { EMPTY, Observable } from 'rxjs';
import { environment } from '../../../environments/environment.development';
import { catchError } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class DatasetUpdatedService {
  private socket$: WebSocketSubject<UserUpdatedPayload> | null = null;
  private readonly WS_ENDPOINT = environment.wsUrl + 'datasets_ws/dataset_updated';

  public connect(): Observable<UserUpdatedPayload> {
    if (!this.socket$ || this.socket$.closed) {
      this.socket$ = webSocket<UserUpdatedPayload>(this.WS_ENDPOINT);
    }
    return this.socket$.pipe(
      catchError((err) => {
        console.error('Connection error WebSocket:', err);
        return EMPTY;
      }),
    );
  }

  public disconnect(): void {
    if (this.socket$) {
      this.socket$.complete();
      this.socket$ = null;
    }
  }
}
