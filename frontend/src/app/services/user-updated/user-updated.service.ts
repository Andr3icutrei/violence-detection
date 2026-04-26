import { Injectable } from '@angular/core';
import { BehaviorSubject, EMPTY, Observable, Subject, tap } from 'rxjs';
import { environment } from '../../../environments/environment.development';
import { webSocket, WebSocketSubject } from 'rxjs/webSocket';
import { catchError } from 'rxjs/operators';

export interface UserUpdatedPayload {
  type: string;
  data: {
    userId: number;
    updatedAt: string;
  };
}

@Injectable({
  providedIn: 'root',
})
export class UserUpdatedService {
  private socket$: WebSocketSubject<UserUpdatedPayload> | null = null;
  private readonly WS_ENDPOINT = environment.wsUrl + 'users_ws/user-updated';

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

  public disconnect(): void {
    if (this.socket$) {
      this.socket$.complete();
      this.socket$ = null;
    }
  }
}
