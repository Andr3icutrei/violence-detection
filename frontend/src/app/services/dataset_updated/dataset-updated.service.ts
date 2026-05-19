import { Injectable } from '@angular/core';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { UserUpdatedPayload } from '../user-updated/user-updated.service';
import { EMPTY, Observable, Subject } from 'rxjs';
import { environment } from '../../../environments/environment.development';
import { catchError } from 'rxjs/operators';

@Injectable({
  providedIn: 'root',
})
export class DatasetUpdatedService {
  private socket$: WebSocketSubject<UserUpdatedPayload> | null = null;
  private readonly WS_ENDPOINT = environment.wsUrl + 'datasets_ws/dataset_updated';
  private readonly localUpdates$ = new Subject<void>();

  public connect(): Observable<UserUpdatedPayload> {
    if (!this.socket$ || this.socket$.closed) {
      this.socket$ = webSocket<UserUpdatedPayload>(this.WS_ENDPOINT);
    }
    return this.socket$.asObservable().pipe(
      catchError((err) => {
        console.error('Connection error WebSocket:', err);
        return EMPTY;
      }),
    );
  }

  public updates(): Observable<void> {
    return this.localUpdates$.asObservable();
  }

  public emitUpdate(): void {
    this.localUpdates$.next();
  }

  public disconnect(): void {
    if (this.socket$) {
      this.socket$.complete();
      this.socket$ = null;
    }
  }
}
