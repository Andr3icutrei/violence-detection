import { Injectable } from '@angular/core';
import { Observable, Subject } from 'rxjs';

@Injectable({
  providedIn: 'root',
})
export class TopbarRefreshService {
  private readonly refreshSubject: Subject<void> = new Subject<void>();

  public get refresh$(): Observable<void> {
    return this.refreshSubject.asObservable();
  }

  public notifyRefresh(): void {
    this.refreshSubject.next();
  }
}

